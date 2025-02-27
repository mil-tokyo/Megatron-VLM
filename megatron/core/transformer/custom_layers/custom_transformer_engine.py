# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
import collections
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from packaging.version import Version as PkgVersion

import torch
import torch.nn.functional as F

try:
    import transformer_engine_extensions as tex
except ImportError:
    import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import (
    cast_to_fp8,
    cast_from_fp8,
)
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.module import LayerNormLinear, Linear
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
    split_tensor_along_dim,
    get_device_compute_capability,
    get_default_init_method,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    AttnBiasTypes,
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
    checkpoint,
    # CudaRNGStatesTracker,
    # graph_safe_rng_available,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.jit import jit_fuser, no_torch_dynamo
# from transformer_engine.pytorch.graph import is_graph_capturing


_flash_attn_version = PkgVersion(get_pkg_version("flash-attn"))
_flash_attn_version_required = PkgVersion("2.0.6")
_flash_attn_max_version = PkgVersion("2.5.8")
_flash_attn_2_1_plus = _flash_attn_version >= PkgVersion("2.1")
_flash_attn_2_3_plus = _flash_attn_version >= PkgVersion("2.3")
_flash_attn_2_4_plus = _flash_attn_version >= PkgVersion("2.4")
_flash_attn_2_4_1_plus = _flash_attn_version >= PkgVersion("2.4.1")

if _flash_attn_version >= _flash_attn_version_required:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func # pylint: disable=no-name-in-module
    from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd # pylint: disable=no-name-in-module
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward # pylint: disable=no-name-in-module,ungrouped-imports
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward # pylint: disable=no-name-in-module


QKVLayouts = (
    "sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
    "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
    "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd")


META_QKV  = tex.FP8FwdTensors.GEMM1_OUTPUT
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1
META_O    = tex.FP8FwdTensors.GEMM2_INPUT
META_DO   = tex.FP8BwdTensors.GRAD_INPUT2
META_S    = tex.FP8FwdTensors.GEMM3_OUTPUT
META_DP   = tex.FP8BwdTensors.GRAD_INPUT3

_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))
_alibi_cache = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
    }


__all__ = ["DotProductAttention", "InferenceParams"]


_ALL_ACTIVE_RNG_STATES = {}

def set_all_rng_states(states: List) -> None:
    """Updates all generator states used by `CudaRNGStatesTracker`."""
    global _ALL_ACTIVE_RNG_STATES
    _ALL_ACTIVE_RNG_STATES = states


class InferenceParams: # pylint: disable=too-few-public-methods
    """
    Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference.

    Parameters
    ----------
    max_batch_size : int
                    maximum batch size during inference.
    max_sequence_length : int
                         maximum sequence length during inference.
    """

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_indices):
        """
        Reorders the KV cache using the specified batch indices.

        Parameters
        ----------
        batch_indices : List[int]
                       Sequence of indices to reorder along the batch dimensions of
                       the KV cache. Must have a length equal to the batch size.
        """
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number, inference_memory in self.key_value_memory_dict.items():
            inference_key_memory, inference_value_memory = inference_memory
            assert (
                len(batch_indices) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_indices]
            new_inference_value_memory = inference_value_memory[:, batch_indices]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )

@torch.no_grad()
def get_alibi(
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    bias_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    num_heads: int
        Number of heads.
    max_seqlen_q: int
        Maximum sequence length for queries.
    max_seqlen_kv: int
        Maximum sequence length for keys and values.
    alibi_slopes: Optional[torch.Tensor], default = `None`
        Custom ALiBi slopes, FP32, CUDA tensor, in shape [num_heads] or [batch_size, num_heads].
    bias_dtype: Optional[torch.dtype], default = `None`
        Dtype of the generated ALiBi bias. If None, use torch.float32.

    Returns
    ----------
    alibi_slopes: torch.Tensor
        ALiBi slopes in FP32 and shape [num_heads] or [batch_size, num_heads].
    alibi_bias: torch.Tensor
        ALiBi bias in FP32 or `bias_dtype`. If `alibi_slopes` is in [num_heads] shape,
        then `alibi_bias` is in [1, num_heads, max_seqlen_q, max_seqlen_kv], and if
        `alibi_slopes` is in [batch_size, num_heads], then the bias is in
        [batch_size, num_heads, max_seqlen_q, max_seqlen_kv].
    """
    global _alibi_cache
    if _alibi_cache["_alibi_slopes_require_update"]:
        if alibi_slopes is not None:
            _alibi_cache["_alibi_slopes"] = alibi_slopes
        else:
            n = 2 ** math.floor(math.log2(num_heads))
            m_0 = 2.0 ** (-8.0 / n)
            m = torch.pow(m_0, torch.arange(1, 1 + n))

            if n < num_heads:
                m_hat_0 = 2.0 ** (-4.0 / n)
                m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
                m = torch.cat([m, m_hat])

            _alibi_cache["_alibi_slopes"] = m.to(dtype=torch.float32, device="cuda")
        _alibi_cache["_num_heads"] = num_heads
        _alibi_cache["_alibi_slopes_require_update"] = False

    if _alibi_cache["_alibi_bias_require_update"]:
        assert _alibi_cache["_alibi_slopes"] is not None, "ALiBi slopes can not be None!"
        if _alibi_cache["_alibi_slopes"].dim() == 1:
            slopes_shape = torch.Size([1, _alibi_cache["_alibi_slopes"].shape[0], 1, 1])
        if _alibi_cache["_alibi_slopes"].dim() == 2:
            slopes_shape = torch.Size([*_alibi_cache["_alibi_slopes"].shape[:], 1, 1])
        bias = torch.arange(
            1 - max_seqlen_kv, 1, dtype=torch.int32, device="cuda").view(1, 1, 1, max_seqlen_kv)
        bias = bias - torch.arange(
            1 - max_seqlen_q, 1, dtype=torch.int32, device="cuda").view(1, 1, max_seqlen_q, 1)
        bias = bias.abs().mul(-1)
        bias = bias * _alibi_cache["_alibi_slopes"].view(slopes_shape)
        _alibi_cache["_max_seqlen_q"], _alibi_cache["_max_seqlen_kv"] = max_seqlen_q, max_seqlen_kv
        bias_dtype = torch.float32 if bias_dtype is None else bias_dtype
        _alibi_cache["_alibi_bias"] = bias.contiguous().to(dtype=bias_dtype, device="cuda")
        _alibi_cache["_alibi_bias_require_update"] = False

    return _alibi_cache["_alibi_slopes"], _alibi_cache["_alibi_bias"]


def get_cu_seqlens(mask: torch.Tensor) -> torch.Tensor:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch.
    """
    mask = mask.squeeze(1).squeeze(1)
    reduced_mask = mask.logical_not().sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    return cu_seqlens


def get_cu_seqlens_and_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch, and another int32 tensor of shape [batch_size * max_seqlen, 1, 1]
    containing the indices for the valid tokens.
    """
    mask = mask.squeeze(1).squeeze(1)
    bs, seqlen = mask.shape

    reduced_mask = mask.logical_not().sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    mask = mask.reshape(-1)
    indices = mask.logical_not().nonzero()
    indices = indices.unsqueeze(-1)

    num_nonzeros = indices.shape[0]
    pad_amount = bs * seqlen - num_nonzeros
    indices = F.pad(input=indices, pad=(0, 0, 0, 0, 0, pad_amount),
                    mode="constant", value=float(bs * seqlen))

    return cu_seqlens, indices


def get_indices(max_seqlen: int, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Given max_seqlen and cu_seqlens of shape [batch_size + 1], returns an int32
    tensor of shape [batch_size * max_seqlen, 1, 1] containing the indices for
    the valid tokens in a batch.
    """
    bs = len(cu_seqlens) - 1
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = [i*max_seqlen + ii for i,j in enumerate(seqlens) for ii in range(j)]
    indices = torch.Tensor(indices).unsqueeze(1).unsqueeze(1).to(
                    dtype=torch.int64, device="cuda")

    num_nonzeros = indices.shape[0]
    pad_amount = bs * max_seqlen - num_nonzeros
    indices = F.pad(input=indices, pad=(0, 0, 0, 0, 0, pad_amount),
                    mode="constant", value=float(bs * max_seqlen))

    return indices

_cu_seqlens_cache = {}
def _get_full_cu_seqlens(
    batch_size: int,
    max_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Cumulative sequence lengths in full data batch

    All sequences in batch have the maximum sequence length.

    """
    global _cu_seqlens_cache
    if (batch_size, max_seqlen) not in _cu_seqlens_cache:
        _cu_seqlens_cache[(batch_size, max_seqlen)] = torch.arange(
            0,
            (batch_size + 1) * max_seqlen,
            step=max_seqlen,
            dtype=torch.int32,
            device=device,
        )
    return _cu_seqlens_cache[(batch_size, max_seqlen)]


@jit_fuser
def pack_tensor(
    indices: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Packs the given tensor using the `indices`.
    """
    padding_indice = torch.zeros(
        1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat((tensor, padding_indice), dim=0)

    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    packed = torch.gather(tensor, 0, indices)
    return packed


@jit_fuser
def pack_2_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs the given 2 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    return t1_packed, t2_packed


@jit_fuser
def pack_3_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs the given 3 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    t3_packed = pack_tensor(indices, t3)
    return t1_packed, t2_packed, t3_packed


@jit_fuser
def unpack_tensor(
    indices: torch.Tensor,
    dim0: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of `pack_tensor`.
    """
    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    unpacked = torch.zeros(
        dim0 + 1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
    unpacked.scatter_(0, indices, tensor)
    unpacked = unpacked[0:-1,:,:]
    return unpacked


@jit_fuser
def unpack_2_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_2_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    return t1_unpacked, t2_unpacked


@jit_fuser
def unpack_3_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_3_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    t3_unpacked = unpack_tensor(indices, dim0, t3)
    return t1_unpacked, t2_unpacked, t3_unpacked


class PackTensors(torch.autograd.Function):
    """
    Autograd function to pack tensors.
    """
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        *tensors: Tuple[torch.Tensor, ...]
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        assert 1 <= len(tensors) <= 3, f"Packing {len(tensors)} tensors not supported."
        ctx.save_for_backward(indices)
        ctx.dim0 = tensors[0].shape[0]
        if len(tensors) == 1:
            return pack_tensor(indices, *tensors)
        if len(tensors) == 2:
            return pack_2_tensors(indices, *tensors)
        return pack_3_tensors(indices, *tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tuple[torch.Tensor, ...]):
        (indices,) = ctx.saved_tensors
        if len(grad_outputs) == 1:
            return None, unpack_tensor(indices, ctx.dim0, *grad_outputs)
        if len(grad_outputs) == 2:
            return None, *unpack_2_tensors(indices, ctx.dim0, *grad_outputs)
        return None, *unpack_3_tensors(indices, ctx.dim0, *grad_outputs)


class UnpackTensor(torch.autograd.Function):
    """
    Autograd function to unpack a tensor.
    """
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        dim0: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices)
        return unpack_tensor(indices, dim0, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return None, None, pack_tensor(indices, grad_output)


def flash_attn_p2p_communicate(rank, send_tensor, send_dst,
                               recv_tensor, recv_src,
                               cp_group, batch_p2p_comm):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@jit_fuser
def flash_attn_fwd_out_correction(out, out_per_step, seq_dim,
                                  softmax_lse, softmax_lse_per_step):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step*softmax_lse_corrected_exp
    out.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


class AttnFuncWithCP(torch.autograd.Function):
    """
    Attention implementation with context parallelism.
    Split attention compute into multiple steps, and overlap current-step
    compute with next-step communication.
    """

    @staticmethod
    def forward(ctx, is_training, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, cp_group, cp_global_ranks, cp_stream, softmax_scale, qkv_format,
                attn_mask_type, attn_bias_type, attn_bias, deterministic, use_fused_attention):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank - 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        causal = (attn_mask_type == "causal")

        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [x.view(x.shape[0], 2, x.shape[1]//2, *x.shape[2:]) for x in [q, k, v]]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0]//2, *x.shape[1:]) for x in [q, k, v]]
        if attn_bias is not None:
            assert (len(attn_bias.shape) == 4), (
                "Only support bias shape of [b, h, sq, sk] for forward, "
                "and [1, h, sq, sk] for backward!"
            )
            # [b, np, sq, sk] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_bias_ = attn_bias.view( \
                *attn_bias.shape[:-2], \
                2, attn_bias.shape[-2]//2, \
                2*cp_size, attn_bias.shape[-1]//(2*cp_size) \
            )
            # [b, np, sq, sk] -> [b, np, sq, 2*cp, sk//(2*cp)]
            attn_bias = attn_bias.view( \
                *attn_bias.shape[:-1], \
                2*cp_size, attn_bias.shape[-1]//(2*cp_size) \
            )
        assert(q.shape[-1] % 8 == 0), "hidden size per attention head should be multiple of 8"
        fa_optional_forward_kwargs = {}
        if _flash_attn_2_3_plus:
            fa_optional_forward_kwargs["window_size"] = [-1, 0] if causal else [-1, -1]
        if _flash_attn_2_4_plus:
            fa_optional_forward_kwargs["alibi_slopes"] = None

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        attn_bias_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        for i in range(cp_size+1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i%2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i+1)%2]:
                        req.wait()

                    if i < (cp_size-1):
                        p2p_comm_buffers[i+1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i%2] = flash_attn_p2p_communicate(rank,
                                                                         p2p_comm_buffers[i],
                                                                         send_dst,
                                                                         p2p_comm_buffers[i+1],
                                                                         recv_src,
                                                                         cp_group,
                                                                         batch_p2p_comm)

                    kv_inputs[i%2] = p2p_comm_buffers[i]
                    if causal:
                        if i == 0:
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                    q_inputs[i%2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2].view(
                                        2, k.shape[0], -1, *k.shape[-2:])
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                    q_inputs[i%2] = q.view(-1, *q.shape[-3:])
                                    # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2].view(
                                        2, -1, *k.shape[-3:])
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i%2] = torch.cat(
                                        (attn_bias[..., idx, :], \
                                         attn_bias[..., (2*cp_size-idx-1), :]),
                                        dim=-1
                                    ).contiguous()
                                out_per_step[i], [softmax_lse_per_step[i], rng_states[i], *rest] = \
                                fused_attn_fwd(
                                    is_training, max_seqlen_q, max_seqlen_k, cu_seqlens_q,
                                    cu_seqlens_k, q_inputs[i%2], kv_inputs[i%2][0],
                                    kv_inputs[i%2][1], TE_DType[q.dtype],
                                    tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                                    attn_scale=softmax_scale, dropout=dropout_p,
                                    qkv_layout=qkv_layout, attn_mask_type="causal",
                                    attn_bias_type=attn_bias_type, attn_bias=attn_bias_inputs[i%2],
                                )
                                if len(rest) > 0:
                                    attn_biases[i] = rest[0]
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                    dropout_p, softmax_scale, causal=True, return_softmax=False,
                                    **fa_optional_forward_kwargs
                                )
                        elif i <= rank:
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                    q_inputs[i%2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2][:, :, 0, ...].contiguous()
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                    q_inputs[i%2] = q.view(-1, *q.shape[-3:])
                                    # [2, 2, sk//2, b, np, hn] -> [2, sk//2, b, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2][:, 0, ...].contiguous()
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i%2] = attn_bias[..., idx, :].contiguous()
                                out_per_step[i], [softmax_lse_per_step[i], rng_states[i], *rest] = \
                                fused_attn_fwd(
                                    is_training, max_seqlen_q, max_seqlen_k//2, cu_seqlens_q,
                                    cu_seqlens_k//2, q_inputs[i%2], kv_inputs[i%2][0],
                                    kv_inputs[i%2][1], TE_DType[q.dtype],
                                    tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                                    attn_scale=softmax_scale, dropout=dropout_p,
                                    qkv_layout=qkv_layout, attn_mask_type="no_mask",
                                    attn_bias_type=attn_bias_type, attn_bias=attn_bias_inputs[i%2],
                                )
                                if len(rest) > 0:
                                    attn_biases[i] = rest[0]
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                                if qkv_format == "thd":
                                    # [2, t, np, hn] -> [2, t/2, np, hn]
                                    kv_inputs[i%2] = tex.thd_read_half_tensor(
                                        kv_inputs[i%2], cu_seqlens_k, 0)
                                else:
                                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2][:, :, 0, ...].contiguous()
                                # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                                kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                                if _flash_attn_2_3_plus:
                                    fa_optional_forward_kwargs["window_size"] = [-1, -1]
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q, cu_seqlens_k//2, max_seqlen_q, max_seqlen_k//2,
                                    dropout_p, softmax_scale, causal=False, return_softmax=False,
                                    **fa_optional_forward_kwargs
                                )
                        else:
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                                    q_inputs[i%2] = q[:, 1, ...].contiguous()
                                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2].view(
                                        2, k.shape[0], -1, *k.shape[-2:])
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                                    q_inputs[i%2] = q[1].contiguous()
                                    # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                                    kv_inputs[i%2] = kv_inputs[i%2].view(
                                        2, -1, *k.shape[-3:])
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i%2] = torch.cat(
                                        (attn_bias_[..., 1, :, idx, :], \
                                         attn_bias_[..., 1, :, (2*cp_size-idx-1), :]),
                                        dim=-1
                                    ).contiguous()
                                out_per_step[i], [softmax_lse_per_step[i], rng_states[i], *rest] = \
                                fused_attn_fwd(
                                    is_training, max_seqlen_q//2, max_seqlen_k, cu_seqlens_q//2,
                                    cu_seqlens_k, q_inputs[i%2], kv_inputs[i%2][0],
                                    kv_inputs[i%2][1], TE_DType[q.dtype],
                                    tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                                    attn_scale=softmax_scale, dropout=dropout_p,
                                    qkv_layout=qkv_layout, attn_mask_type="no_mask",
                                    attn_bias_type=attn_bias_type, attn_bias=attn_bias_inputs[i%2],
                                )
                                if len(rest) > 0:
                                    attn_biases[i] = rest[0]
                            else:
                                if qkv_format == "thd":
                                    # [t, np, hn] -> [t/2, np, hn]
                                    q_inputs[i%2] = tex.thd_read_half_tensor(q, cu_seqlens_q, 1)
                                else:
                                    # [b, 2, sq//2, np, hn]->[b, sq//2, np, hn]->[b*sq//2, np, hn]
                                    q_inputs[i%2] = \
                                        q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                                if _flash_attn_2_3_plus:
                                    fa_optional_forward_kwargs["window_size"] = [-1, -1]
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q//2, cu_seqlens_k, max_seqlen_q//2, max_seqlen_k,
                                    dropout_p, softmax_scale, causal=False, return_softmax=False,
                                    **fa_optional_forward_kwargs
                                )
                    else:
                        if use_fused_attention:
                            if attn_bias is not None:
                                idx = (rank - i) % cp_size
                                attn_bias_inputs[i%2] = torch.cat(
                                    (attn_bias[..., idx, :], attn_bias[..., (2*cp_size-idx-1), :]),
                                    dim=-1
                                ).contiguous()
                            out_per_step[i], [softmax_lse_per_step[i], rng_states[i], *rest] = \
                            fused_attn_fwd(
                                is_training, max_seqlen_q, max_seqlen_k, cu_seqlens_q,
                                cu_seqlens_k, q, kv_inputs[i%2][0],
                                kv_inputs[i%2][1], TE_DType[q.dtype],
                                tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                                attn_scale=softmax_scale, dropout=dropout_p,
                                qkv_layout=qkv_layout, attn_mask_type="no_mask",
                                attn_bias_type=attn_bias_type, attn_bias=attn_bias_inputs[i%2],
                            )
                            if len(rest) > 0:
                                attn_biases[i] = rest[0]
                        else:
                            # [b, sq, np, hn] -> [b*sq, np, hn]
                            q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            _, _, _, _, out_per_step[i], \
                            softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                dropout_p, softmax_scale, causal=False, return_softmax=False,
                                **fa_optional_forward_kwargs
                            )

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i-1)%2].wait_event(fwd_results_correction_done)

                if use_fused_attention:
                    # [b, np, sq, 1] -> [b, np, sq]
                    softmax_lse_per_step[i-1].squeeze_(-1)

                with torch.cuda.stream(flash_attn_streams[(i-1)%2]):
                    if i == 1:
                        out = torch.empty_like(q).zero_()
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(torch.double)
                        if causal and qkv_format != "thd":
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2
                            )
                    elif (i-1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(softmax_lse,
                                                              softmax_lse_per_step[i-1])
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(softmax_lse,
                                                               softmax_lse_per_step[i-1],
                                                               cu_seqlens_q,
                                                               q.size(0))
                        else:
                            flash_attn_fwd_softmax_lse_correction(softmax_lse_[..., 1, :],
                                                                  softmax_lse_per_step[i-1])

                if i < cp_size:
                    flash_attn_streams[(i-1)%2].record_event(fwd_results_correction_done)

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
        for i in range(cp_size):
            if qkv_format == "bshd":
                out_per_step[i] = out_per_step[i].view(out.shape[0], -1, *out.shape[-2:])
                out_ = out[:, 1, ...]
            elif qkv_format == "sbhd":
                out_per_step[i] = out_per_step[i].view(-1, *out.shape[-3:])
                out_ = out[1]

            if i <= rank or not causal:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_out_correction(out.view(*out_per_step[i].shape),
                                                  out_per_step[i],
                                                  seq_dim,
                                                  softmax_lse,
                                                  softmax_lse_per_step[i])
                elif qkv_format == "thd":
                    tex.thd_out_correction(out,
                                           out_per_step[i],
                                           softmax_lse,
                                           softmax_lse_per_step[i],
                                           cu_seqlens_q,
                                           False)
                else:
                    assert False, f"{qkv_format} is an unsupported qkv_format!"
            else:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_out_correction(out_,
                                                  out_per_step[i],
                                                  seq_dim,
                                                  softmax_lse_[..., 1, :],
                                                  softmax_lse_per_step[i])
                elif qkv_format == "thd":
                    tex.thd_out_correction(out,
                                           out_per_step[i],
                                           softmax_lse,
                                           softmax_lse_per_step[i],
                                           cu_seqlens_q,
                                           True)
                else:
                    assert False, f"{qkv_format} is an unsupported qkv_format!"

        kv = p2p_comm_buffers[-1]
        if use_fused_attention:
            if qkv_format == "bshd":
                out = out.view(out.shape[0], -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                out = out.view(-1, *out.shape[-3:])
        else:
            out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(q, kv, out, softmax_lse,
            cu_seqlens_q, cu_seqlens_k, *rng_states, *attn_biases)
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.qkv_format = qkv_format
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k) = ctx.saved_tensors[:6]
        cp_size = get_distributed_world_size(ctx.cp_group)
        rng_states = ctx.saved_tensors[6:6+cp_size]
        attn_biases = ctx.saved_tensors[6+cp_size:6+cp_size*2]

        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        if attn_biases[0] is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)]
            attn_dbias = torch.zeros(
                *ctx.attn_bias_shape,
                dtype=attn_biases[0].dtype,
                device=attn_biases[0].device
            )
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_dbias_ = attn_dbias.view(
                *attn_dbias.shape[:-3], 2, attn_dbias.shape[-3]//2, *attn_dbias.shape[-2:]
            )
        else:
            attn_dbias = None

        if ctx.causal:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(softmax_lse, cu_seqlens_q, q.size(0))
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = \
                    softmax_lse.view(*softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2)
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
                if ctx.use_fused_attention:
                    # [b, np, sq//2] -> [b, np, sq//2, 1]
                    softmax_lse_.unsqueeze_(-1)

        if ctx.use_fused_attention:
            # [b, np, sq] -> [b, np, sq, 1]
            softmax_lse.unsqueeze_(-1)
        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        # Flash Attn outputs
        dq = torch.empty_like(q)

        p2p_comm_buffers = [torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device), \
                            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)]
        p2p_comm_buffers[0][0].copy_(kv)
        send_recv_reqs = []

        fa_optional_backward_kwargs = {}
        if _flash_attn_2_4_plus:
            fa_optional_backward_kwargs["alibi_slopes"] = None
        if _flash_attn_2_4_1_plus:
            fa_optional_backward_kwargs["deterministic"] = ctx.deterministic

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i%2]
            recv_tensor = p2p_comm_buffers[(i+1)%2]
            if i == 0:
                send_tensor = send_tensor[0]
                recv_tensor = recv_tensor[0]
            if i == (cp_size-1):
                send_tensor = send_tensor[1]
                recv_tensor = recv_tensor[1]

            send_recv_reqs = flash_attn_p2p_communicate(rank,
                                                        send_tensor,
                                                        send_dst,
                                                        recv_tensor,
                                                        recv_src,
                                                        ctx.cp_group,
                                                        batch_p2p_comm)

            kv = p2p_comm_buffers[i%2][0]
            # In reversed order of fwd
            if ctx.causal:
                if i == (cp_size-1):
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            q_ = q.view(q.shape[0], -1, *q.shape[-2:])
                            # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                            kv_ = kv.view(*kv.shape[0:2], -1, *kv.shape[-2:])
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            out_ = out.view(out.shape[0], -1, *out.shape[-2:])
                            dout_ = dout.view(dout.shape[0], -1, *dout.shape[-2:])
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            q_ = q.view(-1, *q.shape[-3:])
                            # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            out_ = out.view(-1, *out.shape[-3:])
                            dout_ = dout.view(-1, *dout.shape[-3:])
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size-i-1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size-i-1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q, ctx.max_seqlen_k,
                            cu_seqlens_q, cu_seqlens_k,
                            q_, kv_[0], kv_[1], out_, dout_,
                            TE_DType[q.dtype], TE_DType[kv.dtype], aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="causal",
                            attn_bias_type=ctx.attn_bias_type,
                        )
                    else:
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        dq_ = torch.empty_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = [-1, 0]
                        _flash_attn_backward(
                            dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                            dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k,
                            ctx.max_seqlen_q, ctx.max_seqlen_k,
                            ctx.dropout_p, ctx.softmax_scale, True,
                            rng_state=rng_states[cp_size-i-1],
                            **fa_optional_backward_kwargs
                        )
                elif i >= (cp_size-rank-1):
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            q_ = q.view(q.shape[0], -1, *q.shape[-2:])
                            # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn]
                            kv_ = kv[:, :, 0, ...].contiguous()
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            out_ = out.view(out.shape[0], -1, *out.shape[-2:])
                            dout_ = dout.view(dout.shape[0], -1, *dout.shape[-2:])
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            q_ = q.view(-1, *q.shape[-3:])
                            # [2, 2, sk//2, b, np, hn] -> [2, sk//2, b, np, hn]
                            kv_ = kv[:, 0, ...].contiguous()
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            out_ = out.view(-1, *out.shape[-3:])
                            dout_ = dout.view(-1, *dout.shape[-3:])
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size-i-1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size-i-1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q, ctx.max_seqlen_k//2,
                            cu_seqlens_q, cu_seqlens_k//2,
                            q_, kv_[0], kv_[1], out_, dout_,
                            TE_DType[q.dtype], TE_DType[kv.dtype], aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                        )
                    else:
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        dq_ = torch.empty_like(q_)
                        if ctx.qkv_format == "thd":
                            # [2, t, np, hn] -> [2, t/2, np, hn]
                            kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_k, 0)
                        else:
                            # [2, b, 2, sk//2, np, hn]->[2, b, sk//2, np, hn]->[2, b*sk//2, np, hn]
                            kv_ = kv[:, :, 0, ...].contiguous().view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = [-1, -1]
                        _flash_attn_backward(
                            dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                            dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k//2,
                            ctx.max_seqlen_q, ctx.max_seqlen_k//2,
                            ctx.dropout_p, ctx.softmax_scale, False,
                            rng_state=rng_states[cp_size-i-1],
                            **fa_optional_backward_kwargs
                        )
                else:
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                            q_ = q[:, 1, ...].contiguous()
                            # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                            kv_ = kv.view(*kv.shape[0:2], -1, *kv.shape[-2:])
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                            out_ = out[:, 1, ...].contiguous()
                            dout_ = dout[:, 1, ...].contiguous()
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                            q_ = q[1].contiguous()
                            # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                            out_ = out[1].contiguous()
                            dout_ = dout[1].contiguous()
                        aux_ctx_tensors = [softmax_lse_, rng_states[cp_size-i-1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size-i-1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q//2, ctx.max_seqlen_k,
                            cu_seqlens_q//2, cu_seqlens_k,
                            q_, kv_[0], kv_[1], out_, dout_,
                            TE_DType[q.dtype], TE_DType[kv.dtype], aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                        )
                    else:
                        if ctx.qkv_format == "thd":
                            # [t, np, hn] -> [t/2, np, hn]
                            q_ = tex.thd_read_half_tensor(q, cu_seqlens_q, 1)
                        else:
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            q_ = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                        dq_ = torch.empty_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        if ctx.qkv_format == "thd":
                            out_ = tex.thd_read_half_tensor(out, cu_seqlens_q, 1)
                            dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q, 1)
                        else:
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            out_ = out[:, 1, ...].contiguous().view(-1, *out.shape[-2:])
                            dout_ = dout[:, 1, ...].contiguous().view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = [-1, -1]
                        _flash_attn_backward(
                            dout_, q_, kv_[0], kv_[1], out_, softmax_lse_,
                            dq_, dkv_[0], dkv_[1], cu_seqlens_q//2, cu_seqlens_k,
                            ctx.max_seqlen_q//2, ctx.max_seqlen_k,
                            ctx.dropout_p, ctx.softmax_scale, False,
                            rng_state=rng_states[cp_size-i-1],
                            **fa_optional_backward_kwargs
                        )
            else:
                if ctx.use_fused_attention:
                    aux_ctx_tensors = [softmax_lse, rng_states[cp_size-i-1]]
                    if attn_dbias is not None:
                        aux_ctx_tensors += [attn_biases[cp_size-i-1]]
                    dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                        ctx.max_seqlen_q, ctx.max_seqlen_k,
                        cu_seqlens_q, cu_seqlens_k,
                        q, kv[0], kv[1], out, dout,
                        TE_DType[q.dtype], TE_DType[kv.dtype], aux_ctx_tensors,
                        tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                        attn_scale=ctx.softmax_scale,
                        dropout=ctx.dropout_p,
                        qkv_layout=qkv_layout,
                        attn_mask_type="no_mask",
                        attn_bias_type=ctx.attn_bias_type,
                    )
                else:
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    q_ = q.view(-1, *q.shape[-2:])
                    dq_ = torch.empty_like(q_)
                    # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                    kv_ = kv.view(2, -1, *kv.shape[-2:])
                    dkv_ = torch.empty_like(kv_)
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    out_ = out.view(-1, *out.shape[-2:])
                    dout_ = dout.view(-1, *dout.shape[-2:])
                    if _flash_attn_2_3_plus:
                        fa_optional_backward_kwargs["window_size"] = [-1, -1]
                    _flash_attn_backward(
                        dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                        dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k,
                        ctx.max_seqlen_q, ctx.max_seqlen_k,
                        ctx.dropout_p, ctx.softmax_scale, False,
                        **fa_optional_backward_kwargs
                    )

            if i >= (cp_size-rank-1) or not ctx.causal:
                # [b*sq, np, hn] -> [b, 2, sq//2, np, hn] if causal
                # [b*sq, np, hn] -> [b, sq, np, hn] if not causal
                dq_ = dq_.view(*dq.shape)
            else:
                if ctx.qkv_format == "bshd":
                    # [b*sq//2, np, hn] -> [b, sq//2, np, hn]
                    dq_ = dq_.view(dq.shape[0], *dq.shape[2:])
                elif ctx.qkv_format == "sbhd":
                    # [b*sq//2, np, hn] -> [sq//2, b, np, hn]
                    dq_ = dq_.view(-1, *dq.shape[-3:])

            if ctx.causal:
                if i > (cp_size-rank-1):
                    dq.add_(dq_)
                elif i == (cp_size-rank-1):
                    if rank == (cp_size-1):
                        dq.copy_(dq_)
                    else:
                        if ctx.qkv_format == "bshd":
                            dq[:, 0, ...].copy_(dq_[:, 0, ...])
                            dq[:, 1, ...].add_(dq_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dq[0].copy_(dq_[0])
                            dq[1].add_(dq_[1])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dq, dq_, cu_seqlens_q, "copy", "add")
                elif i > 0:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].add_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q, "none", "add")
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q, "none", "copy")
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            if attn_dbias is not None:
                idx = (rank+i+1)%cp_size
                if i == (cp_size - 1) or not ctx.causal:
                    # [b, np, sq, sk//cp] -> [b, np, sq, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1]//2)
                    attn_dbias[..., idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias[..., (2*cp_size-idx-1), :].copy_(dbias_[..., 1, :])
                elif i >= (cp_size-rank-1):
                    # [b, np, sq, sk//(2*cp)]
                    attn_dbias[..., idx, :].copy_(dbias_)
                else:
                    # [b, np, sq//2, sk//cp] -> [b, np, sq//2, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1]//2)
                    attn_dbias_[..., 1, :, idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias_[..., 1, :, (2*cp_size-idx-1), :].copy_(dbias_[..., 1, :])

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            dkv = p2p_comm_buffers[(i+1)%2][1]
            if ctx.use_fused_attention:
                dkv_ = torch.cat((dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0)
            if ctx.causal and i >= (cp_size-rank-1) and i != (cp_size-1):
                if ctx.qkv_format == "bshd":
                    # [2, b*sk//2, np, hn] -> [2, b, sk//2, np, hn]
                    dkv_ = dkv_.view(*dkv.shape[0:2], *dkv.shape[3:])
                elif ctx.qkv_format == "sbhd":
                    # [2, b*sk//2, np, hn] -> [2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(dkv.shape[0], -1, *dkv.shape[-3:])
            else:
                # [2, b*sk, np, hn] -> [2, b, 2, sk//2, np, hn] if causal
                # [2, b*sk, np, hn] -> [2, b, sk, np, hn] if not causal
                dkv_ = dkv_.view(*dkv.shape)

            if ctx.causal:
                if i == (cp_size-1):
                    if rank == 0:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                            dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_[:, 0, ...])
                            dkv[:, 1, ...].copy_(dkv_[:, 1, ...])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_k, "add", "copy")
                    else:
                        dkv.add_(dkv_)
                elif i >= (cp_size-rank-1):
                    if i == 0 and rank == (cp_size-1):
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_k, "copy", "none")
                    else:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_k, "add", "none")
                elif i > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                if i == 0:
                    dkv.copy_(dkv_)
                else:
                    dkv.add_(dkv_)

        if ctx.causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(q.shape[0], -1, *q.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                dkv = dkv.view(*kv.shape[0:2], -1, *kv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *q.shape[-3:])
                # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                dkv = dkv.view(kv.shape[0], -1, *kv.shape[-3:])

        if attn_dbias is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, sq, sk]
            attn_dbias = attn_dbias.view(*attn_dbias.shape[:-2], -1)

        return None, dq, dkv[0], dkv[1], None, None, None, None, None, None, \
                None, None, None, None, None, None, attn_dbias, None, None


def attn_forward_func_with_cp(
    is_training, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p, cp_group, cp_global_ranks, cp_stream, softmax_scale=None, qkv_format="bshd",
    attn_mask_type="causal", attn_bias_type="no_bias", attn_bias=None, deterministic=False,
    use_fused_attention=False
) -> torch.Tensor:
    """Attention implementation with context parallelism"""
    assert(qkv_format in ["bshd", "sbhd", "thd"]
        ), f"QKV format of {qkv_format} is not supported with context parallelism!"
    assert(qkv_format != "sbhd" or use_fused_attention
        ), "FlashAttention does not support sbhd format!"
    assert(not(qkv_format == "thd" and use_fused_attention)
        ), "FusedAttention does not support thd format!"
    assert (attn_mask_type in ["causal", "no_mask"]
        ), f"Mask type of {attn_mask_type} is not supported with context parallelism!"
    assert (attn_bias is None or use_fused_attention
        ), "Attention bias is only supported with FusedAttention!"
    out = AttnFuncWithCP.apply(
        is_training, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, cp_group, cp_global_ranks, cp_stream, softmax_scale, qkv_format,
        attn_mask_type, attn_bias_type, attn_bias, deterministic, use_fused_attention
    )
    return out


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """
    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        rotary_percent: float
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int
            if not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int
            pre-trained max_position_embeddings before position interpolation
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )
        self.register_buffer('inv_freq', inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if (self.pretrained_max_position_embeddings is not None
            and self.seq_len_interpolation_factor is not None):
            if (max_seq_len >
                self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum('i , j -> i j', seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


class UnfusedDotProductAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(attention_mask_func)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # An FP16 training trick required for certain GPT-like models.
        self.apply_qk_layer_scaling = (
            bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and layer_number is not None)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None, # pylint: disable=unused-argument
        cu_seqlens_kv: Optional[torch.Tensor] = None, # pylint: disable=unused-argument
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unfused attention fprop"""
        def print_value(key_name, value_tensor: torch.Tensor, hf_value: float, disable=True):
            if not disable:
                value_pad_len = 10
                hf_pad_len = 10
                if value_tensor.mean().item() < 0:
                    value_pad_len -= 1
                if hf_value < 0:
                    hf_pad_len -= 1
                print(f"{key_name:<17}{value_tensor.mean().item():.{value_pad_len}f} | hf: {hf_value:.{hf_pad_len}f}")

        print_value("query_layer", query_layer, -0.07303236424922943)
        print_value("key_layer", key_layer, 0.011032702401280403)
        print_value("value_layer", value_layer, -0.002397199161350727)

        assert (qkv_layout in QKVLayouts
            ), f"UnfusedDotProductAttention does not support qkv_layout = {qkv_layout}!"
        qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
        assert (qkv_format != 'thd'
            ), """UnfusedDotProductAttention does not support variable sequence lengths!"""
        if qkv_format == 'bshd':
            # convert to sbhd and use sbhd implementation for now
            query_layer, key_layer, value_layer = [x.transpose(0, 1)
                for x in [query_layer, key_layer, value_layer]]

        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if key_layer.shape[2] != query_layer.shape[2]:
            assert (query_layer.shape[2]%key_layer.shape[2]==0
                ),"The number of attention heads must be divisible by the number of GQA groups!"
            key_layer = key_layer.repeat_interleave(
                    int(query_layer.shape[2]/key_layer.shape[2]), dim = 2)
            value_layer = value_layer.repeat_interleave(
                    int(query_layer.shape[2]/value_layer.shape[2]), dim = 2)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        # WAR to set dtype to FP32 as ONNX lacks BF16 support for ConstantOfShape operator
        is_bf16 = query_layer.dtype == torch.bfloat16
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=torch.float32 if is_in_onnx_export_mode() and is_bf16 else query_layer.dtype,
            device=torch.cuda.current_device(),
        )
        matmul_result = torch.zeros_like(matmul_result)

        if is_in_onnx_export_mode() and is_bf16:
            matmul_result = matmul_result.bfloat16()

        scale = self.norm_factor
        if apply_qk_layer_scaling:
            scale *= self.layer_number

        # Raw attention scores. [b * np, sq, sk]
        if core_attention_bias_type == "no_bias":
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / scale),
            )
            print_value("matmul_result", matmul_result, -1.106471061706543)

        elif core_attention_bias_type == "pre_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            matmul_result = torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            )
            matmul_result = (matmul_result.view(
                output_size[0], output_size[1], output_size[2], output_size[3])
                + core_attention_bias).view(-1, output_size[2], output_size[3])
            matmul_result /= scale

        elif core_attention_bias_type in ["post_scale_bias", "alibi"]:
            if core_attention_bias_type == "post_scale_bias":
                assert core_attention_bias is not None, "core_attention_bias should not be None!"
            if core_attention_bias_type == "alibi":
                _, core_attention_bias = get_alibi(
                    output_size[1], output_size[2], output_size[3], alibi_slopes=alibi_slopes)
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / scale),
            )
            matmul_result = (matmul_result.view(
                output_size[0], output_size[1], output_size[2], output_size[3])
                + core_attention_bias).view(-1, output_size[2], output_size[3]).to(
                dtype=query_layer.dtype)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        softmax_scale = self.layer_number if apply_qk_layer_scaling else None
        attention_probs = self.scale_mask_softmax(
            attention_scores, attention_mask, attn_mask_type, softmax_scale
        )
        torch.cuda.synchronize()
        print_value("attention_probs", attention_probs, 0.001733102253638208)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        # ここでずれてる？
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        torch.cuda.synchronize()
        print_value("context_layer", context_layer, -0.002489551203325391)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        if qkv_format == 'sbhd':
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.view(seqlen, batch_size, -1)

        if qkv_format == 'bshd':
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [b, sq, hp]
            context_layer = context_layer.view(batch_size, seqlen, -1)

        return context_layer


class _PrepareQKVForFA(torch.autograd.Function):
    """This class converts QKV from interleaved (s, b, ...) layout
       to separate contiguous q, k, v tensors in (b, s, ...) layout."""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # All inputs received are non-contiguous tensors.
        # The `query_layer` tensor is used to access the
        # full memory region of the QKV tensor.
        qkv = tex.fa_prepare_fwd(query_layer)
        q, k, v = split_tensor_along_dim(qkv, 0, 3)
        query_layer = torch.squeeze(q, 0)
        key_layer = torch.squeeze(k, 0)
        value_layer = torch.squeeze(v, 0)
        return query_layer, key_layer, value_layer

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        dqkv = tex.fa_prepare_bwd(dq, dk, dv)
        dq, dk, dv = split_tensor_along_dim(dqkv, -1, 3)
        return dq, dk, dv


def _get_qkv_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_format: str = 'sbhd',
    ) -> str:
    """Get qkv layout.

    Parameters
    ----------
    q: torch.Tensor
        Query tensor.
    k: torch.Tensor
        Key tensor.
    v: torch.Tensor
        Value tensor.
    qkv_format: str, default = `sbhd`
        Dimension format for `q`, `k` and `v`, {`sbhd`, `bshd`, `thd`}. `s` stands for
        the sequence length dimension, `b` batch size, `h` the number of attention heads,
        `d` head size, and `t` the total number of sequences in a batch, i.e.
        `t = sum(s_i) for i = 0...b-1`.

    Returns
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. Each `qkv_format` can be mapped to one of five
       memory layouts. For example, `sb3hd` means `q`, `k`, `v` are created as one chunk
       of memory and that they are interleaved in the `2`nd dimension. `sbhd_sbh2d` means
       `q` and `kv` are created in two chunks and that `q` itself is contiguous and `k`, `v`
       are interleaved with each other in the `3`rd dimension, `k = kv[:,:,:,0,:]` and
       `v = kv[:,:,:,1,:]`.
       Mapping:
       `sbhd`: {`sb3hd`, `sbh3d`, `sbhd_sb2hd`, `sbhd_sbh2d`, `sbhd_sbhd_sbhd`}
       `bshd`: {`bs3hd`, `bsh3d`, `bshd_bs2hd`, `bshd_bsh2d`, `bshd_bshd_bshd`}
       `thd` : {`t3hd`, `th3d`, `thd_t2hd`, `thd_th2d`, `thd_thd_thd`}
    """

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"

    def run_iteratively(q, k, v):
        data_ptr = q.untyped_storage().data_ptr()
        check_ptrs_qkv = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
        data_ptr = k.untyped_storage().data_ptr()
        check_ptrs_kv = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])

        stride = q.stride()
        check_strides_qkv = all(stride == x.stride() for x in [q, k, v])
        stride = k.stride()
        check_strides_kv = all(stride == x.stride() for x in [k, v])

        shape = q.shape
        check_shapes_qkv = all(shape == x.shape for x in [q, k, v])
        shape = k.shape
        check_shapes_kv = all(shape == x.shape for x in [k, v])

        last_dim_size = q.shape[-1]
        check_last_dim_offsets_qkv = all(i * last_dim_size == x.storage_offset()
                            for i, x in enumerate([q, k, v]))
        last_dim_size = k.shape[-1]
        check_last_dim_offsets_kv = all(i * last_dim_size == x.storage_offset()
                            for i, x in enumerate([k, v]))

        last_two_dims_size = q.shape[-1] * q.shape[-2]
        check_last_two_dims_offsets_qkv = all(i * last_two_dims_size == x.storage_offset()
                            for i, x in enumerate([q, k, v]))
        last_two_dims_size = k.shape[-1] * k.shape[-2]
        check_last_two_dims_offsets_kv = all(i * last_two_dims_size == x.storage_offset()
                            for i, x in enumerate([k, v]))

        if (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
            and check_last_two_dims_offsets_qkv
            and not check_last_dim_offsets_qkv):
            # sb3hd, bs3hd, t3hd
            qkv_layout = qkv_format[:-2] + '3' + qkv_format[-2:]
        elif (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
            and check_last_dim_offsets_qkv):
            # sbh3d, bsh3d, th3d
            qkv_layout = qkv_format[:-1] + '3' + qkv_format[-1:]
        elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
            and check_last_two_dims_offsets_kv
            and not check_last_dim_offsets_kv):
            # sbhd_sb2hd, bshd_bs2hd, thd_t2hd
            qkv_layout = qkv_format + '_' + qkv_format[:-2] + '2' + qkv_format[-2:]
        elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
            and check_last_dim_offsets_kv):
            # sbhd_sbh2d, bshd_bsh2d, thd_th2d
            qkv_layout = qkv_format + '_' + qkv_format[:-1] + '2' + qkv_format[-1:]
        elif check_strides_kv and check_shapes_kv:
            # sbhd_sbhd_sbhd, bshd_bshd_bshd, thd_thd_thd
            qkv_layout = '_'.join(list([qkv_format])*3)
        else:
            qkv_layout = 'not_supported'

        return qkv_layout

    qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == 'not_supported':
        # force q,k,v to be contiguous and run get_layout again
        q, k, v = [x.contiguous() for x in [q, k, v]]
        qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == 'not_supported':
        raise Exception("The provided qkv memory layout is not supported!")

    return qkv_layout, q, k, v


def check_set_window_size(
        attn_mask_type: str,
        window_size: Tuple[int, int] = None,
    ):
    """Check if sliding window size is compliant with mask type and if not,
    assert or set it to the appropriate size
    """
    if "causal" in attn_mask_type:
        if window_size is None:
            window_size = (-1, 0)
        else:
            assert (
                window_size[1] == 0
            ), "window_size[1] should be 0 when self_attn_mask_type includes 'causal'!"
    else:
        if window_size is None:
            window_size = (-1, -1)
    return window_size


class FlashAttention(torch.nn.Module):
    """Dot product attention, using HazyResearch flash-attn package:
    https://github.com/Dao-AILab/flash-attention
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        assert (
            _flash_attn_version >= _flash_attn_version_required
        ), f"FlashAttention minimum version {_flash_attn_version_required} is required."
        assert (
            _flash_attn_version <= _flash_attn_max_version
        ), f"FlashAttention maximum version {_flash_attn_max_version} is supported."

        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> torch.Tensor:
        """flash-attn fprop"""

        window_size = check_set_window_size(attn_mask_type, window_size)

        assert (
            query_layer.dtype in [torch.float16, torch.bfloat16]
            and key_layer.dtype in [torch.float16, torch.bfloat16]
            and value_layer.dtype in [torch.float16, torch.bfloat16]
            ), "FlashAttention currently only supports FP16 and BF16."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), "FlashAttention currently only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
            ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

        context_parallel = (cp_group is not None) and (get_distributed_world_size(cp_group) != 1)

        qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])

        if qkv_format == 'sbhd':
            # For now just 128, will make it more general in the future
            if (query_layer.shape[-1] == 128 and
                query_layer.shape[0] * query_layer.shape[1] >= 512 and
                qkv_layout == "sbh3d"):
                query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(query_layer,
                                                                             key_layer,
                                                                             value_layer)
            else:
                query_layer, key_layer, value_layer = [x.transpose(0,1).contiguous()
                    for x in (query_layer, key_layer, value_layer)]
        elif qkv_format == 'bshd':
            query_layer, key_layer, value_layer = [x.contiguous()
                for x in (query_layer, key_layer, value_layer)]

        batch_size = query_layer.shape[0]

        if qkv_format in ['sbhd', 'bshd']:
            max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
            if not context_parallel:
                # [b * s, h, d]
                query_layer, key_layer, value_layer = [
                    x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

            if 'padding' in attn_mask_type:
                assert not context_parallel, "Padding mask not supported with context parallelism!"

                if self.attention_type == "self":
                    assert (
                        max_seqlen_q == max_seqlen_kv
                    ), "Maximum sequence length for Q and KV should be the same."
                    if cu_seqlens_q is None:
                        assert (attention_mask is not None
                                ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask)
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                    cu_seqlens_kv = cu_seqlens_q
                    query_layer, key_layer, value_layer = PackTensors.apply(
                        indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        assert (attention_mask is not None
                            ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(
                            attention_mask[0])
                        cu_seqlens_kv, indices_kv = get_cu_seqlens_and_indices(
                            attention_mask[1])
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                        indices_kv = get_indices(max_seqlen_kv, cu_seqlens_kv)
                    query_layer = PackTensors.apply(indices_q, query_layer)
                    key_layer, value_layer = PackTensors.apply(
                        indices_kv, key_layer, value_layer
                    )
            else:
                # Cumulative sequence lengths for unpadded data
                if cu_seqlens_q is None:
                    cu_seqlens_q = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )
        elif qkv_format == 'thd':
            assert (cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()

        if context_parallel:
            assert (
                window_size in ((-1, -1), (-1, 0))
                ), "Sliding window attention is not supported with context parallelism."
            assert (
                alibi_slopes is None
            ), "Alibi slope bias addition is not supported with context parallelism."
            with self.attention_dropout_ctx():
                output = attn_forward_func_with_cp(
                    self.training, query_layer, key_layer, value_layer,
                    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                    self.attention_dropout if self.training else 0.0,
                    cp_group, cp_global_ranks, cp_stream,
                    softmax_scale=1.0/self.norm_factor,
                    qkv_format="bshd" if qkv_format=="sbhd" else qkv_format,
                    attn_mask_type=attn_mask_type,
                    deterministic=self.deterministic
                )
        else:

            from .cpu_offload import CPUOffloadEnabled
            if CPUOffloadEnabled:
                tensor_list = [query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv]
                for tensor in tensor_list:
                    if tensor is not None:
                        tensor.activation_offloading = True

            with self.attention_dropout_ctx():
                fa_optional_forward_kwargs = {}
                if _flash_attn_2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = window_size
                if _flash_attn_2_4_plus:
                    fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
                if _flash_attn_2_4_1_plus:
                    fa_optional_forward_kwargs["deterministic"] = self.deterministic
                output = flash_attn_forward_func(
                    query_layer, key_layer, value_layer,
                    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                    self.attention_dropout if self.training else 0.0,
                    softmax_scale=1.0/self.norm_factor, causal="causal" in attn_mask_type,
                    **fa_optional_forward_kwargs,
                )

        if 'padding' in attn_mask_type:
            output = UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)

        if qkv_format == 'sbhd':
            # (bs)hd -> bs(hd) -> sb(hd)
            output = output.view(batch_size, max_seqlen_q, -1).transpose(0, 1).contiguous()
        elif qkv_format == 'bshd':
            # (bs)hd -> bs(hd)
            output = output.view(batch_size, max_seqlen_q, -1).contiguous()
        elif qkv_format == 'thd':
            # thd -> t(hd)
            output = output.view(output.shape[0], -1).contiguous()

        return output


class DotProductAttention(torch.nn.Module):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    .. warning::

        FlashAttention uses a non-deterministic algorithm for optimal performance. To observe
        deterministic behavior at the cost of performance, use FlashAttention version >= `2.4.1`
        and set the environment variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order
        to disable`flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : int
                number of key-query-value channels per attention head.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation, options are "`no_mask`",
                   "`padding`", "`causal`", "`padding,causal`", "`causal,padding`", and
                   "`arbitrary`", where "`padding,causal`" and "`causal,padding`" are equivalent.
                   This arg can be overridden by :attr:`attn_mask_type` in the `forward` method.
                   It is useful for cases involving compilation/tracing, e.g. ONNX export, and the
                   forward arg is useful for dynamically changing mask types, e.g. a different mask
                   for training and inference. For "`no_mask`", no attention mask is applied. For
                   "`causal`" or the causal mask in "`padding,causal`", TransformerEngine calculates
                   and applies an upper triangular mask to the softmax input. No user input is
                   needed. For "`padding`" or the padding mask in "`padding,causal`", users need to
                   provide the locations of padded tokens either via :attr:`cu_seqlens_q` and
                   :attr:`cu_seqlens_kv` in the shape of [batch_size + 1] or :attr:`attention_mask`
                   in the shape [batch_size, 1, 1, max_seq_len]. For the "`arbitrary`" mask, users
                   need to provide a mask that is broadcastable to the shape of softmax input.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Similar to :attr:`attn_mask_type`, it can
                be overridden by :attr:`window_size` in `forward` as well.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length, `b` batch size,
               `h` the number of heads, `d` head size, and `t` the total number of sequences
               in a batch, with `t = sum(s_i), for i = 0...b-1`. `sbhd` and `bshd` formats
               are used for when sequences in a batch are of equal length or padded to
               equal length, and the `thd` format is used for when sequences in a batch
               have different lengths. Please note that these formats do not reflect how
               tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
               For that, please use `_get_qkv_layout` to gain the layout information.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : ProcessGroup, default = `None`
              context parallel process group.
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism splits flash attention into multiple steps for
               compute and communication overlapping. To address the wave quantization
               issue of each split step, we add an additional CUDA stream so that we
               can overlap two flash attention kernels.
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> None:
        super().__init__()

        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",","_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type
        self.window_size = window_size
        self.window_size = check_set_window_size(attn_mask_type, self.window_size)
        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_group = tp_group
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream

        self.hidden_size_per_attention_head = kv_channels

        self.num_gqa_groups = (
            num_attention_heads if num_gqa_groups is None else num_gqa_groups
        )
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)

        assert (num_attention_heads % self.num_gqa_groups == 0
                ), "The number of attention heads must be divisible by the number of GQA groups!"

        self.rng_states_tracker = None
        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            self.rng_states_tracker = get_rng_state_tracker()
            set_all_rng_states(self.rng_states_tracker.get_states())
            attention_dropout_ctx = self.rng_states_tracker.fork

        norm_factor = math.sqrt(kv_channels)

        self.device_compute_capability = get_device_compute_capability()
        self.deterministic = not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))) \
                             or torch.are_deterministic_algorithms_enabled()

        # self.use_flash_attention = (
        #     int(os.getenv("NVTE_FLASH_ATTN", "1"))
        #     # and self.device_compute_capability >= (8, 0)
        # )
        # if not _flash_attn_2_4_1_plus and self.deterministic:
        #     self.use_flash_attention = False
        #     warnings.warn(
        #         "Disabling usage of FlashAttention since version <2.4.1 does not support "
        #         "deterministic execution. In order to use FA with deterministic behavior,"
        #         " please install FlashAttention version >=2.4.1."
        #     )
        self.use_flash_attention = False

        self.use_fused_attention = (
            int(os.getenv("NVTE_FUSED_ATTN", "1"))
            # and self.device_compute_capability >= (8, 0)
        )

        assert (
            attention_type in AttnTypes
        ), f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
        }

        # if self.use_flash_attention:
        #     self.flash_attention = FlashAttention(norm_factor,
        #                                           attention_type=attention_type,
        #                                           layer_number=layer_number,
        #                                           deterministic=self.deterministic,
        #                                           **attn_kwargs)

        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        # if self.use_fused_attention:
        #     self.fused_attention = FusedAttention(norm_factor,
        #                                           attention_type=attention_type,
        #                                           layer_number=layer_number,
        #                                           deterministic=self.deterministic,
        #                                           **attn_kwargs)

        self.unfused_attention = UnfusedDotProductAttention(
            norm_factor, **attn_kwargs, layer_number=layer_number)

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        hidden_states = checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=self.get_rng_state_tracker,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

        return hidden_states

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : ProcessGroup
                  context parallel process group.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        """
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream

    @no_torch_dynamo(recursive=False)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_format: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        inference_params: Optional[InferenceParams] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes '"padding"' or `"arbitrary"`.

        .. note::

            Input tensor :attr:`query_layer` must be of shape
            (:attr:`sequence_length`, :attr:`batch_size`, :attr:`num_attention_heads`,
            :attr:`kv_channels`) and the tensors :attr:`key_layer` and :attr:`value_layer`
            must each be of shape (:attr:`sequence_length`, :attr:`batch_size`,
            :attr:`num_gqa_groups`, :attr:`kv_channels`). Output of shape
            (:attr:`sequence_length`, :attr:`batch_size`, :attr:`num_attention_heads`
            * :attr:`kv_channels`) is returned.

        .. note::

            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

        .. note::

            Users can use environment variables :attr:`NVTE_FLASH_ATTN`, :attr:`NVTE_FUSED_ATTN`,
            and :attr:`NVTE_FUSED_ATTN_BACKEND` to control which DotProductAttention backend,
            and FusedAttention backend if applicable, to use. TransformerEngine prioritizes
            FlashAttention over FusedAttention and over UnfusedDotProductAttention.
            If FusedAttention is being used, users can also choose to switch to flash-attn's
            implementation for backward by setting :attr:`NVTE_FUSED_ATTN_USE_FAv2_BWD=1`
            (default: 0), because of the performance differences between various versions of
            flash-attn and FusedAttention. Further, :attr:`NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT`
            can be used to enable (:attr:`1`) or disable (:attr:`0`) the workspace related
            optimizations in FusedAttention. When unset, TransformerEngine determines the code path
            based on its internal logic. These optimizations trade memory for performance
            and should be used with care.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be 'None' for 'causal' and 'no_mask' types. For 'padding' masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For the 'arbitrary' mask type, it should be in a shape that is
             broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value
             means the corresponding position is masked out and a `False` means that position is
             allowed to participate in attention.
        qkv_format: str, default = `None`
                   If provided, overrides :attr:`qkv_format` from initialization.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths in a batch for `key_layer` and `value_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      Calculated from `cu_seqlens_q` if not provided.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       Calculated from `cu_seqlens_kv` if not provided.
        attn_mask_type: {`no_mask`, `padding`, `causal`, `padding,causal`, `causal,padding`,
                       `arbitrary`}, default = `None`. Type of attention mask passed into
                       softmax operation. 'padding,causal' and 'causal,padding' are equivalent.
        window_size: Optional[Tuple[int, int]], default = `None`
                    Sliding window size for local attention.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        inference_params: Optional[InferenceParams], default = `None`
            Optimizes execution performance during inference by caching Keys and Values of the
            current decoding iteration. These cached values are appended to the K and V values
            computed in previous iterations, eliminating the need to recalculate them for the
            entire sequence.
            Initialization of `inference_params` is required prior to use to ensure sufficient
            memory allocation.
            Adjustments of the sequence_len_offset should be done after a complete forward pass.
            If rotary positional embeddings (RoPE) are utilized, they must be prepared beforehand.
            Supports "sbhd" and "bshd" layouts, with the "sbhd" layout being more efficient.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), 'DotProductAttention only supports CUDA tensors.'

        assert (key_layer.shape == value_layer.shape
            ), "Keys and values must have the same shape!"

        if attn_mask_type is not None:
            window_size = check_set_window_size(attn_mask_type, window_size)
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        else:
            attn_mask_type = attn_mask_type.replace(",","_")
            if attn_mask_type == "causal_padding":
                attn_mask_type = "padding_causal"

        assert (attn_mask_type in AttnMaskTypes
            ), f"Attention mask type {attn_mask_type} is not supported!"

        # if self.rng_states_tracker is not None and is_graph_capturing():
        #     assert (
        #         isinstance(self.rng_states_tracker, CudaRNGStatesTracker)
        #     ), "Unsupported RNG states tracker."
        #     assert (
        #         graph_safe_rng_available()
        #     ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

        if window_size is None:
            window_size = self.window_size

        if qkv_format is None:
            qkv_format = self.qkv_format

        if inference_params is not None:
            assert self.layer_number is not None, "Layer number must be set!"

            if qkv_format == "bshd":
                key_layer = key_layer.transpose(0, 1)
                value_layer = value_layer.transpose(0, 1)

            (inference_key_memory, inference_value_memory,
            ) = inference_params.key_value_memory_dict[self.layer_number]

            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)

            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)

            # Copy keys and values into KV-cache
            inference_key_memory[
                sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[
                sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

            if qkv_format == "bshd":
                key_layer = key_layer.transpose(0, 1)
                value_layer = value_layer.transpose(0, 1)

            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        assert (key_layer.shape[-2] == self.num_gqa_groups_per_partition
            and value_layer.shape[-2] == self.num_gqa_groups_per_partition
            ), f"Keys and values must have num_gqa_group = {self.num_gqa_groups} heads!"
        assert (qkv_format in ['sbhd', 'bshd', 'thd']
            ), "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"

        if qkv_format == 'thd':
            assert (all(len(x.shape) == 3 for x in (query_layer, key_layer, value_layer))
                ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
            assert (cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            assert (cu_seqlens_q.shape == cu_seqlens_kv.shape
                and len(cu_seqlens_q.shape) == 1
                and len(cu_seqlens_kv.shape) == 1
                ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
            assert (cu_seqlens_q.dtype == torch.int32
                and cu_seqlens_kv.dtype == torch.int32
                ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()

        if qkv_format in ['sbhd', 'bshd']:
            assert (all(len(x.shape) == 4 for x in (query_layer, key_layer, value_layer))
                ), f"Queries, keys and values must be 4D tensors when qkv_format = {qkv_format}!"
            if qkv_format == 'sbhd':
                max_seqlen_q, max_seqlen_kv = (query_layer.shape[0], key_layer.shape[0])
            if qkv_format == 'bshd':
                max_seqlen_q, max_seqlen_kv = (query_layer.shape[1], key_layer.shape[1])
            if cu_seqlens_q is not None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                assert (all(seqlens_q <= max_seqlen_q)
                    ), """Sequence lengths indicated by cu_seqlens_q must be no greater than
                    the sequence dimention in 'query_layer'!"""
            if cu_seqlens_kv is not None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                assert (all(seqlens_kv <= max_seqlen_kv)
                    ), """Sequence lengths indicated by cu_seqlens_kv must be no greater than
                    the sequence dimention in 'key_layer' and 'value_layer'!"""

        qkv_layout, query_layer, key_layer, value_layer = _get_qkv_layout(
            query_layer, key_layer, value_layer, qkv_format = qkv_format)

        # The priority for attention backends (subject to availability and clearing the filters)
        # is: FlashAttention > FusedAttention (cuDNN) > UnfusedDotProductAttention.
        use_flash_attention = self.use_flash_attention
        use_fused_attention = self.use_fused_attention
        use_unfused_attention = True

        # The following section filters out some backends based on
        # certain asserts before executing the forward pass.

        # Filter: ONNX export.
        if is_in_onnx_export_mode():
            use_flash_attention = False
            use_fused_attention = False

        # Filter: Input type.
        # if (query_layer.dtype not in [torch.bfloat16, torch.float16]
        #     or key_layer.dtype not in [torch.bfloat16, torch.float16]
        #     or value_layer.dtype not in [torch.bfloat16, torch.float16]
        #     or any(isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer])
        # ):
        #     use_flash_attention = False
        # if (query_layer.dtype not in [torch.bfloat16, torch.float16]
        #     or key_layer.dtype not in [torch.bfloat16, torch.float16]
        #     or value_layer.dtype not in [torch.bfloat16, torch.float16]
        # ):
        #     use_fused_attention = False
        use_flash_attention = False
        use_fused_attention = False

        # Filter: Device and dimensions.
        # FAv2 supports head_dim <= 256, and for >192 requires sm80/sm90
        # FAv2 requires head_dim % 8 == 0
        if (key_layer.shape[-1] > 256
            or key_layer.shape[-1] % 8 != 0
            or (key_layer.shape[-1] > 192
                # and self.device_compute_capability not in ((8, 0), (9, 0))
                )):
            use_flash_attention = False

        # Filter: cross attention + causal mask.
        # (in training mode)
        if (inference_params is None
            and _flash_attn_2_1_plus
            and "causal" in attn_mask_type
            and max_seqlen_q != max_seqlen_kv
        ):
            warnings.warn(
                "In training mode, disable the use of FlashAttention since version 2.1+ has "
                "changed its behavior for causal mask in cross attention. See "
                "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
            )
            use_flash_attention = False

        context_parallel = (self.cp_group is not None and \
            get_distributed_world_size(self.cp_group) != 1)

        # Filter: sliding window attention.
        # UnfusedDotProductAttention can support SWA via arbitrary attention mask.
        if window_size not in ((-1, -1), (-1, 0)):
            use_fused_attention = False
            if (not _flash_attn_2_3_plus) or context_parallel:
                use_flash_attention = False

        # Filter: Attention mask type.
        #   attn_mask_type(s)    |     supported backends
        # ------------------------------------------------
        #   no_mask              |     All
        #   padding              |     UnfusedDotProductAttention, FlashAttention, FusedAttention
        #   causal               |     All
        #   padding + causal     |     FlashAttention, FusedAttention
        #   arbitrary            |     UnfusedDotProductAttention
        #
        if attn_mask_type == "arbitrary":
            use_flash_attention = False
            use_fused_attention = False

        if (inference_params is None
            and "causal" in attn_mask_type
            and max_seqlen_q != max_seqlen_kv
        ):
            use_unfused_attention = False

        # Filter: bias.
        global _alibi_cache
        if alibi_slopes is not None:
            assert (core_attention_bias_type == "alibi"
                ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
            if self.layer_number == 1:
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True
        if core_attention_bias_type == "alibi":
            assert (core_attention_bias is None
                ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
            if (_alibi_cache["_num_heads"] != query_layer.shape[-2]
                or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                or _alibi_cache["_alibi_slopes"] is None):
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True

        if core_attention_bias_type not in ["no_bias", "alibi"] or core_attention_bias is not None:
            use_flash_attention = False

        fu_core_attention_bias_type = core_attention_bias_type
        fu_core_attention_bias = core_attention_bias
        if core_attention_bias_type == "alibi" and use_fused_attention and alibi_slopes is not None:
            fu_core_attention_bias_type = "post_scale_bias"
            _, fu_core_attention_bias = get_alibi(
                query_layer.shape[-2], max_seqlen_q, max_seqlen_kv, alibi_slopes=alibi_slopes,
                bias_dtype=query_layer.dtype)
        if (use_fused_attention
            and fu_core_attention_bias_type == "post_scale_bias"
            and (fu_core_attention_bias.shape[0] != 1
            or fu_core_attention_bias.shape[1] != query_layer.shape[-2])):
            if fu_core_attention_bias.requires_grad:
                # remove this line when cuDNN adds bwd support for
                # [1, 1, s, s], [b, 1, s, s] and [b, h, s, s]
                use_fused_attention = False
            else:
                # max512 backend will only support [1, h, s, s]
                os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"

        # if use_fused_attention:
        #     fused_attention_backend = tex.get_fused_attn_backend(
        #         TE_DType[query_layer.dtype]
        #         if not isinstance(query_layer, Float8Tensor) else query_layer._fp8_dtype,
        #         TE_DType[key_layer.dtype]
        #         if not isinstance(key_layer, Float8Tensor) else key_layer._fp8_dtype,
        #         QKVLayout[qkv_layout],
        #         AttnBiasType[fu_core_attention_bias_type],
        #         AttnMaskType[attn_mask_type],
        #         self.attention_dropout,
        #         query_layer.shape[-2], # num_attn_heads
        #         key_layer.shape[-2], # num_gqa_groups
        #         max_seqlen_q,
        #         max_seqlen_kv,
        #         query_layer.shape[-1], # head_dim
        #     )
        #     # DPA does not support FP8; for FP8, use cpp_extensions modules directly
        #     is_backend_avail = (fused_attention_backend in
        #         [FusedAttnBackend["F16_max512_seqlen"],
        #         FusedAttnBackend["F16_arbitrary_seqlen"],
        #         FusedAttnBackend["FP8"]])
        #     use_fused_attention = ( \
        #         use_fused_attention and is_backend_avail and \
        #         (not context_parallel or \
        #          fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]))
        #     if (fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]
        #         and fu_core_attention_bias_type == "post_scale_bias"
        #         and (fu_core_attention_bias.shape[0] != 1
        #         or fu_core_attention_bias.shape[1] != query_layer.shape[-2])):
        #         use_fused_attention = False

        # Filter: determinism.
        # backend                                  | deterministic
        # ---------------------------------------------------------
        # flash-attn v1                            | yes
        # flash-attn v2                            | no
        # FusedAttnBackend["F16_max512_seqlen"]    | yes
        # FusedAttnBackend["F16_arbitrary_seqlen"] | workspace optimization path: yes; otherwise: no
        # UnfusedDotProductAttention               | yes
        #
        # Note that FusedAttnBackend["F16_arbitrary_seqlen"] only has workspace optimization path
        # on sm90 architectures.
        #
        # if (use_fused_attention
        #     and fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
        #     and self.deterministic
        #     and self.device_compute_capability != (9, 0)):
        #     use_fused_attention = False

        # Select FusedAttention on sm90 and FlashAttention on others for performance
        # if (use_flash_attention
        #     and use_fused_attention
        #     and fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]):
        #     if self.device_compute_capability == (9, 0):
        #         use_flash_attention = False

        # if use_flash_attention:
        #     if _NVTE_DEBUG:
        #         print("[DotProductAttention]: using flash-attn",_flash_attn_version)
        #     if core_attention_bias_type == "alibi":
        #         alibi_slopes, _ = get_alibi(
        #             query_layer.shape[-2], max_seqlen_q, max_seqlen_kv, alibi_slopes=alibi_slopes)
        #     return self.flash_attention(query_layer,
        #                                 key_layer,
        #                                 value_layer,
        #                                 attention_mask=attention_mask,
        #                                 qkv_layout=qkv_layout,
        #                                 cu_seqlens_q=cu_seqlens_q,
        #                                 cu_seqlens_kv=cu_seqlens_kv,
        #                                 attn_mask_type=attn_mask_type,
        #                                 window_size=window_size,
        #                                 alibi_slopes=alibi_slopes,
        #                                 cp_group=self.cp_group,
        #                                 cp_global_ranks=self.cp_global_ranks,
        #                                 cp_stream=self.cp_stream,
        #                                 max_seqlen_q=max_seqlen_q,
        #                                 max_seqlen_kv=max_seqlen_kv)
        #
        # if use_fused_attention:
        #     if _NVTE_DEBUG:
        #         print("[DotProductAttention]: using cuDNN fused attention (backend "
        #             + str(int(fused_attention_backend)) + ")")
        #     if checkpoint_core_attention:
        #         return self._checkpointed_attention_forward(
        #             self.fused_attention,
        #             query_layer,
        #             key_layer,
        #             value_layer,
        #             qkv_layout=qkv_layout,
        #             cu_seqlens_q=cu_seqlens_q,
        #             cu_seqlens_kv=cu_seqlens_kv,
        #             max_seqlen_q=max_seqlen_q,
        #             max_seqlen_kv=max_seqlen_kv,
        #             attn_mask_type=attn_mask_type,
        #             attention_mask=attention_mask,
        #             fused_attention_backend=fused_attention_backend,
        #             core_attention_bias_type=fu_core_attention_bias_type,
        #             core_attention_bias=fu_core_attention_bias,
        #             fast_zero_fill=fast_zero_fill,
        #             cp_group=self.cp_group,
        #             cp_global_ranks=self.cp_global_ranks,
        #             cp_stream=self.cp_stream,
        #             is_first_microbatch=is_first_microbatch)
        #     return self.fused_attention(
        #         query_layer,
        #         key_layer,
        #         value_layer,
        #         qkv_layout=qkv_layout,
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_kv=cu_seqlens_kv,
        #         max_seqlen_q=max_seqlen_q,
        #         max_seqlen_kv=max_seqlen_kv,
        #         attn_mask_type=attn_mask_type,
        #         attention_mask=attention_mask,
        #         fused_attention_backend=fused_attention_backend,
        #         core_attention_bias_type=fu_core_attention_bias_type,
        #         core_attention_bias=fu_core_attention_bias,
        #         fast_zero_fill=fast_zero_fill,
        #         cp_group=self.cp_group,
        #         cp_global_ranks=self.cp_global_ranks,
        #         cp_stream=self.cp_stream,
        #         is_first_microbatch=is_first_microbatch)

        assert (not context_parallel), \
            "Context parallelism is only implemented with Flash Attention and Fused Attention!"

        # from .cpu_offload import CPUOffloadEnabled
        # if CPUOffloadEnabled:
        #     warnings.warn(
        #                    "Attention activation Offloading is only implemented"
        #                    "with Flash Attention and Fused Attention!"
        #                  )

        if _NVTE_DEBUG:
            print("[DotProductAttention]: using unfused DPA")
        if use_unfused_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(
                    self.unfused_attention,
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout = qkv_layout,
                    cu_seqlens_q = cu_seqlens_q,
                    cu_seqlens_kv = cu_seqlens_kv,
                    attn_mask_type = attn_mask_type,
                    attention_mask = attention_mask,
                    core_attention_bias_type = core_attention_bias_type,
                    core_attention_bias = core_attention_bias,
                    alibi_slopes = alibi_slopes)
            return self.unfused_attention(query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout = qkv_layout,
                    cu_seqlens_q = cu_seqlens_q,
                    cu_seqlens_kv = cu_seqlens_kv,
                    attn_mask_type = attn_mask_type,
                    attention_mask = attention_mask,
                    core_attention_bias_type = core_attention_bias_type,
                    core_attention_bias = core_attention_bias,
                    alibi_slopes = alibi_slopes)

        raise Exception("No dot product attention support for the provided inputs!")
