# GraphMamba periodic delta V3 code archive

This file preserves the retired module contract needed to reconstruct V3. It is
an experiment archive, not active source.

## Exact delta operation

The V3 Mamba-1 path copied the installed non-fused forward up to the delta
projection, then replaced the selective-scan delta preparation with:

```python
dt = module.dt_proj.weight @ dt.t()
dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
delta_bias = module.dt_proj.bias.to(dtype=dt.dtype)[None, :, None]
dt = F.softplus(dt + delta_bias) * delta_scale.to(dtype=dt.dtype)

y = selective_scan_fn(
    x,
    dt,
    -torch.exp(module.A_log.float()),
    B,
    C,
    module.D.float(),
    z=z,
    delta_bias=None,
    delta_softplus=False,
)
```

Thus scaling occurred after positivity enforcement and directly before state
discretization. Input tokens, convolution output, `A`, `B`, and `C` were not
used as substitutes for elapsed time.

`MambaEncoderLayer.forward` and `MambaEncoder.forward` temporarily accepted a
scalar `delta_scale`. When it was `None`, the installed Mamba path was used.
When non-null, both forward and backward Mamba-1 branches used the explicit
path with the same scale. Non-null scaling on Mamba2 raised
`NotImplementedError`.

## GraphMamba integration

The retired modes were:

```python
periodic_delta_mode in {"legacy", "unit", "physical", "learned"}
```

The scale calculation was:

```python
normalized_stride = periodic_scale_descriptors[:, 1]
relative_stride = normalized_stride / normalized_stride[0]

if periodic_delta_mode == "legacy":
    scales = (None, None)
elif periodic_delta_mode == "unit":
    scales = torch.ones_like(relative_stride)
elif periodic_delta_mode == "physical":
    scales = relative_stride
else:
    exponent = torch.tanh(periodic_delta_exponent)
    scales = torch.exp(exponent * torch.log(relative_stride))
```

`periodic_delta_exponent` was one scalar initialized with `torch.zeros(())`.
For local `(stride=2)` and period `(stride=12)` patches under period 24, the
normalized strides were `[2/24, 12/24]`; unit scales were `[1,1]`, fixed
physical scales `[1,6]`, and learned scales
`[1, 6**tanh(periodic_delta_exponent)]`.

The calls immediately before graph fusion were:

```python
local_temporal = encoder(local_tokens, delta_scale=scales[0])
period_temporal = encoder(period_tokens, delta_scale=scales[1])
```

## Reconstruction warning

The active code deliberately does not expose these APIs. Before restoring,
consult `GraphMamba_periodic_delta_v3_design.md` and
`GraphMamba_periodic_delta_v3_validation.md`. Both learned and fixed-global
variants failed the validation gate; reconstruction should only support a
materially different hypothesis.
