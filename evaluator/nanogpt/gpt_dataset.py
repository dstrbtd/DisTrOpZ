import torch
import numpy as np
import os


class NonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for pre-segmented training data (2D tensor).
    Each row is an independent sequence with no continuity between examples.
    Suitable for datasets already divided into fixed-length chunks.
    """

    def __init__(self, data, device):
        assert data.ndim == 2
        self.examples, self.block_size = data.shape

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


class LazyNonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for pre-segmented training data with lazy loading.
    Chunks are cached locally but only loaded into memory when needed.
    Each row is an independent sequence with no continuity between examples.
    """

    def __init__(self, chunk_ids, cache_location, device, max_chunks_in_memory=None):
        self.chunk_ids = chunk_ids
        self.cache_location = cache_location
        self.device = device
        self.max_chunks_in_memory = max_chunks_in_memory

        # Build index mapping from global index to (chunk_id, local_idx)
        self.chunk_sizes = []
        self.chunk_offsets = []
        total_examples = 0

        print(f"Loading {len(chunk_ids)} chunks to determine sizes")
        for chunk_id in chunk_ids:
            cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"Cached chunk file not found: {cache_file}")

            # Load just to get shape, then discard
            chunk_data = np.load(cache_file)
            assert (
                chunk_data.ndim == 2
            ), f"Expected 2D chunk data, got {chunk_data.ndim}D"

            chunk_size = chunk_data.shape[0]
            self.chunk_sizes.append(chunk_size)
            self.chunk_offsets.append(total_examples)
            total_examples += chunk_size

            # Store block_size from first chunk
            if len(self.chunk_sizes) == 1:
                self.block_size = chunk_data.shape[1]

        self.total_examples = total_examples
        self._loaded_chunks = {}  # Cache for loaded chunks
        self._chunk_access_order = []  # Track access order for LRU eviction

        print(
            f"Dataset initialized: {len(chunk_ids)} chunks, {self.total_examples} total examples"
        )

    def __len__(self):
        return self.total_examples

    def _get_chunk_and_local_idx(self, global_idx):
        """Convert global index to (chunk_id, local_idx)"""
        for i, (chunk_id, offset, size) in enumerate(
            zip(self.chunk_ids, self.chunk_offsets, self.chunk_sizes)
        ):
            if global_idx < offset + size:
                local_idx = global_idx - offset
                return chunk_id, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _evict_old_chunks(self):
        """Remove old chunks from memory if we exceed the limit"""
        if self.max_chunks_in_memory is None:
            return

        while len(self._loaded_chunks) > self.max_chunks_in_memory:
            # Remove least recently used chunk
            oldest_chunk = self._chunk_access_order.pop(0)
            if oldest_chunk in self._loaded_chunks:
                del self._loaded_chunks[oldest_chunk]

    def _load_chunk(self, chunk_id):
        """Load chunk data if not already cached in memory"""
        if chunk_id not in self._loaded_chunks:
            # print(f'loading chunk {chunk_id}')
            cache_file = f"{self.cache_location}/chunk_{chunk_id}.npy"
            chunk_data = np.load(cache_file)
            self._loaded_chunks[chunk_id] = (
                torch.from_numpy(chunk_data).to(device=self.device).long()
            )

            # Evict old chunks if necessary
            self._evict_old_chunks()

        # Update access order for LRU
        if chunk_id in self._chunk_access_order:
            self._chunk_access_order.remove(chunk_id)
        self._chunk_access_order.append(chunk_id)

        return self._loaded_chunks[chunk_id]

    def __getitem__(self, idx):
        chunk_id, local_idx = self._get_chunk_and_local_idx(idx)
        chunk_data = self._load_chunk(chunk_id)
        x = chunk_data[local_idx]
        return x[:-1], x[1:]

    def get_memory_info(self):
        """Return information about current memory usage"""
        return {
            "chunks_in_memory": len(self._loaded_chunks),
            "max_chunks_in_memory": self.max_chunks_in_memory,
            "total_chunks": len(self.chunk_ids),
            "loaded_chunk_ids": list(self._loaded_chunks.keys()),
        }


class ContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for continuous token streams (1D tensor).
    Creates examples by sliding a window over the data.
    Preserves context and long-range dependencies in text.
    """

    def __init__(self, data, block_size, device):
        assert data.ndim == 1

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size + 1]
        return x[:-1], x[1:]
