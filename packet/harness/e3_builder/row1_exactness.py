"""E3-owned backend for the native row1 oracle's seven-method contract.

These method bodies match the VM14b-qualified TorchExactBackend exactly.
The retained E2 overlay intentionally remains frozen and lacks gather_rows;
native row1 must never depend on that older E2 backend's smaller interface.
No CUDA owner or tensor is constructed by importing this module.
"""

class TorchRow1Backend:
    def __init__(self):
        import torch
        self.torch = torch

    def snapshot(self, tensor):
        return tensor.detach().cpu().contiguous()

    def equal(self, a, b):
        return a.shape == b.shape and a.dtype == b.dtype and self.torch.equal(a.view(self.torch.uint8), b.view(self.torch.uint8))

    def nbytes(self, a):
        return a.numel() * a.element_size()

    def finite(self, a):
        flat = a.reshape(-1)
        return all((bool(self.torch.isfinite(flat[i:i + 262144]).all()) for i in range(0, flat.numel(), 262144)))

    def differing_bytes(self, a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return None
        x = a.view(self.torch.uint8).reshape(-1)
        y = b.view(self.torch.uint8).reshape(-1)
        return sum((int((x[i:i + 1048576] != y[i:i + 1048576]).count_nonzero()) for i in range(0, x.numel(), 1048576)))

    def gather_reference(self, kv, table, lo, hi, staged):
        physical = table[lo:hi].to(self.torch.int64)
        self.torch.index_select(kv, 0, physical, out=staged)

    def gather_rows(self, source, source_row, rows, staged, arena_row):
        width = source.shape[-1]
        flat = source.reshape(-1, width)
        dst = staged.reshape(-1, width)
        index = self.torch.arange(source_row, source_row + rows, device=flat.device, dtype=self.torch.int64)
        self.torch.index_select(flat, 0, index, out=dst[arena_row:arena_row + rows])
