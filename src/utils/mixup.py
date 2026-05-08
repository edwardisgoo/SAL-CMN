import torch
from typing import Tuple, List


def _derangement(n: int, device: torch.device) -> torch.Tensor:
    """Return a derangement permutation of range(n) on device.
    Ensures perm[i] != i. If n < 2, returns arange(n).
    """
    if n < 2:
        return torch.arange(n, device=device)
    perm = torch.randperm(n, device=device)
    # Fix fixed points by a single cyclic shift if needed
    fixed = perm.eq(torch.arange(n, device=device))
    if fixed.any():
        perm = perm.roll(1)
        fixed = perm.eq(torch.arange(n, device=device))
        if fixed.any():
            idx = torch.nonzero(fixed).squeeze(-1)
            for i in idx.tolist():
                j = (i + 1) % n
                tmp = perm[j].clone()
                perm[j] = perm[i]
                perm[i] = tmp
    return perm


def mixup_batch(
    batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor],
    mixup_ratio: float = 0.5,
    cut_min: float = 0.2,
    cut_max: float = 0.8,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """CSM-style mixup with aligned cuts and deranged partners.

    Batch fields: (utt_ids, inputs[B,T,*], labels[B,Lmax], label_lengths[B])
    For a random subset of size floor(B*mixup_ratio), pick a derangement within
    the subset, sample a single cut ratio r~U[cut_min, cut_max] per sample, and
    cut both input time T and each sample's label length with the same r.
    Concatenate left(src, r) + right(partner, r). Labels are trimmed/padded to Lmax.
    """
    utt_ids, inputs, labels, label_lengths = batch
    B, T = inputs.shape[0], inputs.shape[1]
    if B == 0 or mixup_ratio <= 0:
        return batch

    num_mix = int(B * mixup_ratio)
    if num_mix < 2:
        return batch  # need at least 2 to derange

    device = inputs.device
    Lmax = labels.shape[1]

    # Choose which samples to mix
    perm = torch.randperm(B, device=device)
    mix_idx = perm[:num_mix]
    # keep_idx = perm[num_mix:]  # clones preserve others

    # Partners as a derangement on the mix set
    partner_local = _derangement(num_mix, device)
    src_idx = mix_idx
    shf_idx = mix_idx[partner_local]

    # Sample per-sample cut ratios
    ratios = torch.rand(num_mix, device=device) * (cut_max - cut_min) + cut_min

    # Compute cut indices (time)
    cut_t = (ratios * T).long().clamp_(1, max(T - 1, 1))

    # Label-domain cuts respect per-sample true lengths
    len_src = label_lengths[src_idx].clamp_min(1)
    len_shf = label_lengths[shf_idx].clamp_min(1)
    cut_lbl_src = (ratios * len_src.float()).long()
    cut_lbl_shf = (ratios * len_shf.float()).long()
    # ensure strictly inside (1..len-1)
    cut_lbl_src = torch.min(torch.max(cut_lbl_src, torch.ones_like(cut_lbl_src)), len_src - 1)
    cut_lbl_shf = torch.min(torch.max(cut_lbl_shf, torch.ones_like(cut_lbl_shf)), len_shf - 1)

    # Mix inputs
    mixed_inputs = torch.empty((num_mix, T) + inputs.shape[2:], device=device, dtype=inputs.dtype)
    for i in range(num_mix):
        si = src_idx[i].item()
        pi = shf_idx[i].item()
        c = int(cut_t[i].item())
        mixed_inputs[i] = torch.cat([inputs[si, :c], inputs[pi, c:]], dim=0)

    # Mix labels
    mixed_labels = torch.empty((num_mix, Lmax), device=device, dtype=labels.dtype)
    mixed_label_lengths = torch.empty((num_mix,), device=device, dtype=label_lengths.dtype)
    for i in range(num_mix):
        si = src_idx[i].item()
        pi = shf_idx[i].item()
        c_src = int(cut_lbl_src[i].item())
        c_shf = int(cut_lbl_shf[i].item())
        left = labels[si, :c_src]
        right = labels[pi, c_shf:]
        lbl = torch.cat([left, right], dim=0)
        new_len = min(lbl.size(0), Lmax)
        mixed_label_lengths[i] = new_len
        if lbl.size(0) < Lmax:
            mixed_labels[i] = torch.cat([lbl, labels.new_zeros(Lmax - lbl.size(0))], dim=0)
        else:
            mixed_labels[i] = lbl[:Lmax]

    # Stitch back
    final_inputs = inputs.clone()
    final_labels = labels.clone()
    final_label_lengths = label_lengths.clone()
    final_inputs[src_idx] = mixed_inputs
    final_labels[src_idx] = mixed_labels
    final_label_lengths[src_idx] = mixed_label_lengths

    return utt_ids, final_inputs, final_labels, final_label_lengths
