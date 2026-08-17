# GraphMamba Mamba-internal literature and code audit

Date: 2026-08-14

## Question and decision

Can the next GraphMamba contribution modify the inside of Mamba rather than add
another output-side module?

Yes, but a generic Mamba modification is not enough. The strongest model-local
gap is the depthwise causal convolution inside the shared Mamba. With the
accepted hourly geometry, adjacent local tokens are separated by 2 hours while
adjacent period tokens are separated by 12 hours. The same learned `d_conv=2`
kernel is therefore shared in token coordinates but not in physical-time
coordinates.

The recommended next step is a diagnostic for a **physical-time local operator**:
represent the causal local kernel as a continuous function of elapsed time and
sample the same function at each branch's physical lags before the shared
selective SSM. This is a provisional research direction, not an accepted model
change.

The novelty claim must be narrow. Continuous kernels, sampling-rate-aware SSMs,
multi-delta Mamba, and altered Mamba input receptive fields already exist. The
potentially new combination is a physically sampled local kernel *inside a
shared multi-resolution selective SSM* for periodic forecasting.

## Local implementation boundary

- Environment: `mamba_ssm==2.2.6.post3`.
- Accepted GraphMamba uses Mamba-1, one encoder layer, bidirectional forward and
  backward Mamba instances, and `d_conv=2`.
- Mamba-1 non-fused execution exposes `x`, delta, `B`, `C`, `A`, `D`,
  and the convolution exactly. The archived V3 implementation already
  reproduced the original non-fused output to maximum error `1.19e-7` on CUDA.
- Mamba-2 packs `x/B/C` into its convolution path and normally calls a fused
  chunk scan. Internal intervention is possible but more invasive.
- Current `Mamba.forward()` accepts only the sequence and inference cache. It
  cannot receive branch scale without a local wrapper or custom block.

## Primary-source overlap and code assessment

| Work | Relevant mechanism | Official-code assessment | GraphMamba decision |
|---|---|---|---|
| Mamba / Mamba-2 / Mamba-3 | Input-dependent delta, B and C; local causal convolution; SSD in Mamba-2; newer Mamba-3 dynamics | `state-spaces/mamba` is Apache-2.0 and actively maintained. It is the authoritative implementation. The installed Mamba-1 is the safest modification boundary; upgrading the backbone is a baseline change, not a contribution. | Reuse APIs and equations with attribution. Do not vendor or upgrade before a controlled backbone benchmark. |
| MambaTS | Removes causal convolution and applies dropout before selective-parameter generation | Official MIT repository. The Mamba-1 fork is readable but disables the fused path; the no-convolution branch is not robust for cached stepping. Its Mamba-2 dropout wraps the complete input projection, not only B/C/delta. It is useful as an ablation reference, not a drop-in module. | Add `no temporal conv` and selective-dropout only as controls. Neither is novel here. |
| MambaSL | Makes delta/B/C time variance switchable, permits `d_conv=0`, and scales the input projection receptive field | Official MIT repository with extensive experiment artifacts. The custom block deliberately uses the non-fused selective scan and targets classification. It is a good implementation reference for controlled internal ablations, but not forecasting-ready code to copy. | Use its ablation logic as precedent. Direct B/C/delta switches or a larger projection alone are crowded. |
| ms-Mamba | Parallel Mamba blocks with fixed or learned sampling-rate multipliers | The paper directly overlaps the archived GraphMamba V3 delta route. No author-official code repository was identified from the paper page or targeted GitHub search in this audit. | No-go. GraphMamba's exact global physical delta multiplier already failed its validation gate. |
| Bi-Mamba+ | Adds a forget gate inside Mamba and applies forward/backward scans | The mechanism is already claimed in the paper. No linked official repository was found on its arXiv page in this audit. | No-go as a primary innovation. Current GraphMamba is already bidirectional; adding the published gate would be adaptation. |
| MambaMixer / Chimera | Token-channel selective mixing / genuine two-dimensional state recurrence | MambaMixer's apparent organization repository contained only a README/image and no detected license or implementation. No linked official Chimera code was found on its arXiv page in this audit. | Architecturally large and overlaps the graph/channel axis. Not a safe module import and not a focused next experiment. |
| RCL | Contrastively pretrains one Mamba block to sharpen selectivity | This changes initialization/training rather than the internal recurrence used at inference. Its reported best replacement/freezing policy varies by dataset. | Possible later training study; not the next structural innovation. |
| FlowState | Adjusts an S5 encoder and functional decoder to sampling rate | Apache-2.0 IBM release exists, but it is an S5 foundation model rather than a Mamba patch forecaster. | Important prior art against broad sampling-rate-invariance claims; not directly portable. |
| CKConv / TPCNN | Parameterizes convolutional kernels as functions of continuous time | Establishes that continuous or time-parameterized convolution and evaluation at arbitrary sampling positions are prior art. | Cite as the local-operator foundation. The continuous kernel itself cannot be claimed as new. |

## Why direct B/C modulation is not the first choice

For Mamba-1, the branch adapter already changes the normalized token presented
to `x_proj`, so delta, B, and C are already indirectly branch-conditioned. A
new scale-FiLM on B/C would be more explicit, but without a diagnostic it risks
adding a second parameterization of information that the current adapter can
already express. MambaSL also establishes direct control of B/C/delta variation,
and Chimera/MambaMixer occupy broad multi-axis selectivity claims.

The local convolution has a sharper non-redundancy argument: its weights are
shared before B/C/delta generation, yet the physical lag represented by each
kernel index changes sixfold across the two branches. The existing affine scale
adapter cannot change those kernel indices or their elapsed-time meaning.

## Proposed mechanism, with claim ceiling

Let branch `s` have patch stride `r_s` in physical time. Replace the discrete
depthwise weights `w[j]` by a shared continuous causal kernel

`w_s[j] = k_theta(j * r_s)`,

with normalization over valid causal lags and a fixed physical support chosen
without validation tuning (initially one detected period). Both branches share
`k_theta` and the complete selective state-space core; only the coordinates at
which the local kernel is evaluated differ.

Potential contribution wording, if validated:

> a physical-time-consistent local operator for shared multi-resolution Mamba,
> which samples one continuous causal kernel on branch-specific patch grids
> while preserving a shared selective state-space dynamics.

This is at most a model-level contribution until it demonstrates consistent
validation gains, sampling-rate transfer, and an ablation over discrete
branch-specific kernels.

## Preregistered diagnostic before implementation

No model code should change until a frozen-checkpoint dependency audit is run.

1. Reproduce the accepted fused checkpoint through the explicit Mamba-1 path.
2. Measure local and period branch prediction sensitivity to removing only the
   temporal part of the Mamba convolution while preserving its pointwise path.
3. Measure branchwise convolution contribution norms and their relation to
   residual error on ordered ETTh1/ETTh2 validation samples.
4. Stop if convolution removal is immaterial in both branches or the two branch
   sensitivities are indistinguishable.
5. If the mismatch is supported, train four same-seed validation-only controls:
   accepted shared discrete convolution; no temporal convolution; independent
   discrete branch convolutions; shared physical-time continuous kernel.
6. Advance only if the physical kernel beats the accepted model on both tasks,
   improves macro MSE by at least 0.5%, does not worsen MAE on both tasks, and
   beats the independent-discrete control. Test remains untouched.

The independent-discrete control is essential: without it, any gain could be
caused merely by adding branch-specific capacity rather than physical-time
consistency.

## Sources

- Mamba paper: https://arxiv.org/abs/2312.00752
- Mamba-2 paper: https://arxiv.org/abs/2405.21060
- Mamba-3 paper: https://arxiv.org/abs/2603.15569
- Official Mamba code (Apache-2.0): https://github.com/state-spaces/mamba
- MambaTS paper: https://arxiv.org/abs/2405.16440
- MambaTS code (MIT): https://github.com/XiudingCai/MambaTS-pytorch
- MambaSL paper: https://openreview.net/forum?id=YDl4vqQqGP
- MambaSL code (MIT): https://github.com/yoom618/MambaSL
- ms-Mamba: https://arxiv.org/abs/2504.07654
- Bi-Mamba+: https://arxiv.org/abs/2404.15772
- MambaMixer: https://arxiv.org/abs/2403.19888
- Chimera: https://arxiv.org/abs/2406.04320
- Repetitive Contrastive Learning: https://arxiv.org/abs/2504.09185
- FlowState: https://arxiv.org/abs/2508.05287
- CKConv: https://arxiv.org/abs/2102.02611
- TPCNN: https://arxiv.org/abs/2308.03210

## Audit limitations

- The mounted academic-search MCP was unavailable; the audit used the skill's
  OpenAlex fallback plus direct primary-paper and author-repository checks.
- Search cannot prove global non-existence. “No official code found” means no
  linked author implementation was identified through the paper page and
  targeted GitHub searches on the audit date.
- No third-party source code was copied and the accepted GraphMamba was not
  modified in this phase.
