# [MODULE DOC]
# HELPER: Type aliases for the torch array-API namespace: Array is torch.Tensor, Device
#   is torch.device, DType is torch.dtype. No runtime side effects beyond importing
#   torch.
__all__ = ["Array", "Device", "DType"]

from torch import device as Device, dtype as DType, Tensor as Array
