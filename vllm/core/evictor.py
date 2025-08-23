# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from collections import deque
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from .core.scheduler import Scheduler


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)





# 实现有前向机制的驱逐策略

class ContentHashAwareLRUEvictor(LRUEvictor):
    """
    LRU + content_hash aware eviction:
    - 避免驱逐 waiting list 中未来会被访问的 block
    """
    def __init__(self, waiting: deque, block_size: int):
        super().__init__()
        self.waiting = waiting  # waiting list
        self.block_size = block_size

    def _is_block_in_waiting(self, block_meta: BlockMetaData) -> bool:
        """
        判断当前 block 的 content_hash 是否出现在 waiting list 中
        """
        block_hash = block_meta.content_hash
        if block_hash is None:
            return False
        if block_hash in compute_waiting_blocks_content_hash(self.waiting, self.block_size):
            return True
        return False

    def evict(self) -> Tuple[int, int]:
        """
        驱逐逻辑：LRU 顺序，但跳过 waiting list 中的 block
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if block_id not in self.free_table:
                continue
            block_meta = self.free_table[block_id]

            # 如果 block 在 waiting list 中则跳过
            if self._is_block_in_waiting(block_meta):
                continue

            # 驱逐该 block
            self.free_table.pop(block_id)
            return block_id, content_hash

        # 如果所有 block 都在 waiting list，可以选择妥协驱逐最旧 block
        block_id, content_hash = next(iter(self.free_table.items()))
        self.free_table.pop(block_id)
        return block_id, content_hash



def compute_waiting_blocks_content_hash(waiting: deque, block_size: int) -> List[int]:
    """
    遍历 waiting list 中的所有 sequences，把 token_ids 按 block_size 分块，
    并计算每个 block 的 content_hash。
    
    Args:
        waiting: Deque[SequenceGroup]
        block_size: 每个 block 的 token 数量

    Returns:
        Dict[block_id, content_hash]
    """
    block_hash_list = []
    block_id_counter = 0  # 为每个 block 分配唯一 ID

    for seq_group in waiting:
        seq = seq_group.seqs[0]
        tokens = seq.inputs.prompt_token_ids
        # 按 block_size 分块
        num_tokens = len(tokens)
        start = 0
        prev_block = None
        while start < num_tokens:
            end = min(start + block_size, num_tokens)
            block_tokens = tokens[start:end]

            # 创建 Block（这里只是模拟初始化，计算 content_hash）
            block = PrefixCachingBlock(
                prev_block=prev_block,
                token_ids=block_tokens,
                block_size=block_size,
                allocator=None  # 如果需要可以传 BlockAllocator
            )

            # 计算 content_hash
            h = block.content_hash
            block_hash_list.append(h)

            # 链接前一个 block
            prev_block = block
            block_id_counter += 1
            start = end

    return block_hash_list


def make_evictor(eviction_policy: EvictionPolicy, scheduler=None, block_size: int = 16) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        waiting = scheduler.get_waiting()
        if waiting is not None:
            return ContentHashAwareLRUEvictor(waiting, block_size)
        else:
            return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
