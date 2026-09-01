# Defect Log

## Pass 0 — Initial Plan Review

| issue | reference evidence | planned fix |
|---|---|---|
| prior lower panels use small boxes and dense typography | user comparison against KARMA | enlarge the detail panels and use 26–30 px labels |
| prior diagram leaves large dead zones inside b/c | user feedback | extend inter-module arrows while scaling modules to fill panel height |
| prior vertical labels are visually cramped | user feedback | 90–105 px-wide operators with top-to-bottom reading direction |
| prior b output is crowded near the right boundary | user feedback | reserve a dedicated sum/output lane on the far right |
| prior c contains more wording than needed | user feedback | reduce to AvgPool, Recent state, Shared decoder, Diff., Gate |

## Screenshot Evidence

| pass | screenshot path | capture type | full canvas visible | crop/viewport notes |
|---|---|---|---|---|
| Cycle 1 | `review/cycle1.png` | canvas-only | yes | 1946 px × 1001 px; Draw.io full-page export, no editor/browser chrome |
| Cycle 2 | `review/cycle2b.png` | canvas-only | yes | 1946 px × 1001 px; corrected re-export after transient HTML-label render |
| Cycle 3 | `review/cycle3.png` | canvas-only | yes | 1946 px × 1001 px; full diagram fills more than 95% of image |
| Cycle 4 | `review/cycle4-final.png` | canvas-only | yes | 1946 px × 1001 px; verifies Cycle 3 boundary fixes |
| Red-team final | `review/final-redteam.png` | canvas-only | yes | 1946 px × 1001 px; latest full-canvas evidence |
| User revision 1 | `review/user-revision1.png` | canvas-only | yes | 1946 px × 1001 px; verifies widened RGSP and straight DHC rows |
| User revision 2 | `review/user-revision2.png` | canvas-only | yes | 1946 px × 1001 px; verifies rebalanced DHC output lane |
| User revision 3 final | `review/user-revision3-final.png` | canvas-only | yes | 1946 px × 1001 px; final full-canvas export used for handoff |
| Manuscript page check | `review/manuscript-page4-final.png` | compiled PDF page | yes | 1489 px × 2105 px at 180 dpi; verifies readability at actual `\textwidth` placement |
| User main-flow replacement | `review/user-mainflow-final.png` | canvas-only | yes | 1946 px × 1001 px; export of `主流程图.drawio` after connector repair |
| User main-flow manuscript check | `review/manuscript-page4-user-mainflow.png` | compiled PDF page | yes | 1489 px × 2105 px at 180 dpi; verifies the replacement inside the paper |
| Panel label/title revision | `review/panel-label-title-revision.png` | canvas-only | yes | 1946 px × 1001 px; verifies lower-left b/c labels and centered expanded module names |
| Panel title manuscript check | `review/manuscript-page4-panel-title-revision.png` | compiled PDF page | yes | 1489 px × 2105 px at 180 dpi; confirms full names remain readable at paper width |
| Lowered modules / initials bold | `review/lowered-modules-initials-bold.png` | canvas-only | yes | 1946 px × 1001 px; verifies 25 px lower module groups and initial-only title emphasis |
| Lowered modules manuscript check | `review/manuscript-page4-lowered-initials.png` | compiled PDF page | yes | 1489 px × 2105 px at 180 dpi; confirms title clearance at actual paper width |

## Screenshot Review

Screenshot-driven cycle inventories will be appended below after each rendered export.

## Defect Inventory — Cycle 1

### P0 — Blockers

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C1-01 | Arrow hygiene | `c_e_pool_decoder` | Orange memory-state edge exits the bottom of AvgPool and loops to the decoder top instead of entering its left upper port. | `review/cycle1.png`, panel c |
| C1-02 | Arrow hygiene | `c_e_state_decoder` | Blue recent-state edge exits the bottom of Recent state and loops to the decoder top instead of entering its left lower port. | `review/cycle1.png`, panel c |

### P1 — Visible defects

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C1-03 | Layout | `c_output` | ΔY is below the Gate, contradicting the requested right-side terminal placement. | `review/cycle1.png`, lower right |
| C1-04 | Arrow hygiene | `c_e_gate_output` | Gate-to-output connector bends downward; terminal flow should remain horizontal. | `review/cycle1.png`, lower right |
| C1-05 | Spacing | `b_sum`, `b_output` | Base-output label almost touches the sum node, leaving no readable arrow span. | `review/cycle1.png`, panel b right |
| C1-06 | Spacing | `b_graph`, `b_sum` | Variable graph-to-sum gap is much shorter than the preceding operator gaps. | `review/cycle1.png`, panel b upper path |
| C1-07 | Arrow hygiene | `b_e_sum_output` | Sum-to-output arrowhead is visually merged with the output glyph. | `review/cycle1.png`, panel b right |
| C1-08 | Typography | `c_e_decoder_diff` | Paired-output label is long and crowds the line immediately before Diff. | `review/cycle1.png`, panel c center-right |
| C1-09 | Layout | `c_gate` | Gate sits too near the right boundary, leaving insufficient output lane. | `review/cycle1.png`, panel c right |
| C1-10 | Spacing | `c_diff`, `c_gate` | Diff.-to-Gate spacing is visibly shorter than decoder-to-Diff. spacing. | `review/cycle1.png`, panel c right |
| C1-11 | Typography | `a_delta` | “Correction ΔY” wraps into two competing semantic labels in one card. | `review/cycle1.png`, panel a lower path |
| C1-12 | Layout | `a_title` | Main title is slightly dominant relative to the module labels at the intended paper width. | `review/cycle1.png`, upper left |
| C1-13 | Spacing | `a_output_label` | Forecast label sits closer to the waves than the Input label, creating asymmetric terminal padding. | `review/cycle1.png`, panel a terminals |
| C1-14 | Style coherence | `panel_a` | Overview panel lacks the pale-blue field present in the visual reference; the page reads flatter than intended. | side-by-side reference comparison |

### P2 — Polish

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C1-15 | Typography | `a_e_split_recent` | Xr label background patch is slightly more visible than other edge labels. | `review/cycle1.png` |
| C1-16 | Typography | `a_e_split_distant` | Xd label background patch is slightly more visible than other edge labels. | `review/cycle1.png` |
| C1-17 | Typography | `a_e_rgsp_base` | Yb edge label is close to the Base card arrowhead. | `review/cycle1.png` |
| C1-18 | Typography | `a_e_dhc_delta` | ΔY edge label is close to the correction-card arrowhead. | `review/cycle1.png` |
| C1-19 | Box integrity | `a_rgsp` | RGSP card has generous internal vertical space after the two-line label. | `review/cycle1.png` |
| C1-20 | Box integrity | `a_dhc` | DHC card has generous internal vertical space after the two-line label. | `review/cycle1.png` |
| C1-21 | Spacing | `a_sum`, `a_denorm` | Neutral connector after the sum is shorter than the role-specific incoming connectors. | `review/cycle1.png` |
| C1-22 | Typography | `b_e_decomp_patches` | Seasonal tensor label is located low on the first vertical jog. | `review/cycle1.png` |
| C1-23 | Typography | `b_e_decomp_trend` | Trend tensor label sits very near the shared decomposition stem. | `review/cycle1.png` |
| C1-24 | Typography | `b_e_patches_bimamba` | Patch-state label is slightly crowded between the patch and Mamba blocks. | `review/cycle1.png` |
| C1-25 | Typography | `b_e_bimamba_graph` | State label has less white background padding than neighboring labels. | `review/cycle1.png` |
| C1-26 | Typography | `b_e_graph_sum` | Seasonal forecast label nearly touches the graph block border. | `review/cycle1.png` |
| C1-27 | Spacing | `b_trend` | Trend projection sits slightly left of the visual center of its long output route. | `review/cycle1.png` |
| C1-28 | Box integrity | `c_decoder` | Shared decoder is tall relative to its single vertical label. | `review/cycle1.png` |
| C1-29 | Typography | `c_e_distant_pool` | Compressed-history label is close to the AvgPool arrowhead. | `review/cycle1.png` |
| C1-30 | Typography | `c_e_recent_state` | Recent-vector label is close to the Recent state arrowhead. | `review/cycle1.png` |
| C1-31 | Style coherence | all panels | Dashed border segments are somewhat long/heavy relative to the module strokes. | `review/cycle1.png` |
| C1-32 | Layout | `b_title`, `c_title` | Detail-panel headings could use slightly more top padding from the dashed border. | `review/cycle1.png` |

Warnings from `preflight-final-before-render.json` were reviewed: the spacing warnings compare unrelated rows across panels; sum nodes are semantically required operators; empty dashed containers are intentionally labeled by separate title cells.

## Fix Verification — Cycle 1

| defect id | claimed fix | old screenshot | new screenshot | status |
|---|---|---|---|---|
| C1-01/C1-02 | removed incorrect rotated-port constraints | cycle1 panel c | cycle2b panel c | ⚠️ PARTIAL — box exits are corrected, but the two routes now merge before the decoder |
| C1-03/C1-04 | aligned ΔY horizontally with Gate | cycle1 lower right | cycle2b lower right | ✅ FIXED |
| C1-05/C1-06/C1-07 | redistributed graph/sum/output x positions | cycle1 panel b right | cycle2b panel b right | ⚠️ PARTIAL — readable, but final arrow remains shorter than desired |
| C1-08 | shortened paired-output notation | cycle1 panel c | cycle2b panel c | ✅ FIXED |
| C1-09/C1-10 | shifted Diff. and Gate left | cycle1 panel c | cycle2b panel c | ✅ FIXED |
| C1-11 | simplified output card to ΔY | cycle1 panel a | cycle2b panel a | ✅ FIXED |
| C1-12 | reduced TimeRole title to 38 px | cycle1 upper left | cycle2b upper left | ✅ FIXED |
| C1-13 | increased output-label padding | cycle1 upper right | cycle2b upper right | ✅ FIXED |
| C1-14 | used pale-blue page field with white detail panels | cycle1 full canvas | cycle2b full canvas | ✅ FIXED |

## Defect Inventory — Cycle 2

### P0 — Blockers

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C2-01 | Semantic/arrow hygiene | `c_e_pool_decoder`, `c_e_state_decoder` | Orange and blue state routes merge into one blue arrow before the shared decoder, falsely implying state addition before paired decoding. | `review/cycle2b.png`, panel c |

### P1 — Visible defects

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C2-02 | Arrow hygiene | `c_e_pool_decoder` | Memory route needs a dedicated upper decoder entry. | `review/cycle2b.png` |
| C2-03 | Arrow hygiene | `c_e_state_decoder` | Recent-state route needs a dedicated lower decoder entry. | `review/cycle2b.png` |
| C2-04 | Typography | `c_e_decoder_diff` | `y+ / y−` remains visually long for the decoder-to-Diff. gap. | `review/cycle2b.png` |
| C2-05 | Spacing | `b_e_sum_output` | Sum-to-base-output connector is still shorter than the requested visible arrow length. | `review/cycle2b.png`, panel b right |
| C2-06 | Spacing | `b_graph`, `b_sum`, `b_output` | Three rightmost RGSP elements do not yet follow an even gap rhythm. | `review/cycle2b.png` |
| C2-07 | Spacing | `c_diff`, `c_gate`, `c_output` | Gate width consumes too much of the final output lane; ΔY is close to the border. | `review/cycle2b.png` |
| C2-08 | Typography | `c_e_diff_gate` | δ label sits near the Gate arrowhead instead of in the center of the connector. | `review/cycle2b.png` |

### P2 — Polish

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C2-09 | Layout | canvas background | Pale-blue background remains visible in the gutter between panels. | `review/cycle2b.png` |
| C2-10 | Typography | `a_e_input_split` | X label slightly overlaps the input arrow shaft visually. | `review/cycle2b.png` |
| C2-11 | Typography | `a_e_recent_condition` | Recent-ref. label is slightly left-heavy on the dashed route. | `review/cycle2b.png` |
| C2-12 | Box integrity | `a_rgsp`, `a_dhc` | Major-stage boxes remain wider than their two-line labels require. | `review/cycle2b.png` |
| C2-13 | Spacing | `a_sum`, `a_denorm` | Sum-to-De-norm connector is visually shorter than De-norm-to-output. | `review/cycle2b.png` |
| C2-14 | Typography | `b_e_graph_sum` | Seasonal forecast label is close to the sum circle. | `review/cycle2b.png` |
| C2-15 | Typography | `b_e_trend_sum` | Trend forecast label is close to the long lower route endpoint. | `review/cycle2b.png` |
| C2-16 | Layout | `c_decoder` | Shared decoder could move a few pixels right to balance its left and right gaps. | `review/cycle2b.png` |
| C2-17 | Style coherence | lower panels | Bottom white cards now match the reference, but corner pale-blue wedges are visible outside them. | `review/cycle2b.png` |

## Fix Verification — Cycle 2

| defect id | claimed fix | old screenshot | new screenshot | status |
|---|---|---|---|---|
| C2-01/C2-02/C2-03 | mapped rotated ports to separate upper/lower decoder entries | cycle2b panel c | cycle3 panel c | ✅ FIXED |
| C2-04 | changed paired notation to compact `y+, y−` | cycle2b panel c | cycle3 panel c | ✅ FIXED |
| C2-05/C2-06 | redistributed Variable graph, sum, and base-output positions | cycle2b panel b | cycle3 panel b | ✅ FIXED |
| C2-07/C2-08 | narrowed Gate and recentered the final connector labels | cycle2b panel c right | cycle3 panel c right | ✅ FIXED |

## Defect Inventory — Cycle 3

### P0 — Blockers

None. All required components and connector directions are present, and no arrow crosses a box or text label.

### P1 — Visible defects

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C3-01 | Spacing | `b_output` | Base-output glyph ends only a few pixels before the RGSP panel boundary. | `review/cycle3.png`, panel b right |
| C3-02 | Spacing | `c_output` | ΔY ends only a few pixels before the DHC panel boundary. | `review/cycle3.png`, panel c right |

### P2 — Polish

| id | zone | element | description | evidence |
|---|---|---|---|---|
| C3-03 | Typography | `b_e_graph_sum` | Seasonal-output label occupies most of the graph-to-sum gap. | `review/cycle3.png` |
| C3-04 | Typography | `b_e_trend_sum` | Trend-output label remains close to the lower route endpoint. | `review/cycle3.png` |
| C3-05 | Layout | canvas background | Pale-blue gutters remain around the white detail panels. | `review/cycle3.png` |
| C3-06 | Spacing | `a_sum`, `a_denorm` | The neutral post-sum link is shorter than the De-norm-to-waveform link. | `review/cycle3.png` |
| C3-07 | Box integrity | `c_decoder` | Shared decoder has more vertical padding than the other vertical operators. | `review/cycle3.png` |
| C3-08 | Style coherence | waveforms | Sine waves are intentionally schematic and smoother than the reference traces. | side-by-side reference comparison |

## Fix Verification — Cycle 3

| defect id | claimed fix | old screenshot | new screenshot | status |
|---|---|---|---|---|
| C3-01 | moved RGSP output glyph left and widened its label cell | cycle3 panel b | cycle4-final panel b | ✅ FIXED |
| C3-02 | moved DHC output glyph left and widened its label cell | cycle3 panel c | cycle4-final panel c | ✅ FIXED |

## Red-Team Visual Audit

| id | zone | finding | cells to change | severity | status |
|---|---|---|---|---|---|
| RT-01 | Arrow hygiene | De-norm-to-output edge contains a small 5 px step because the middle output waveform is not center-aligned. | `a_output_wave_*`, `a_output_label` | P1 | fixed |
| RT-02 | Layout | Decoder-to-Diff. row is about 12 px above the shared decoder center, creating a subtle vertical kink. | `c_diff`, `c_gate`, `c_output` | P1 | fixed |
| RT-03 | Spacing | DHC output still leaves only about 10 px right padding. | `c_gate`, `c_output` | P1 | fixed |
| RT-04 | Spacing | RGSP output still leaves less right padding than the left input margin. | `b_output` | P1 | fixed |
| RT-05 | Typography | Seasonal forecast label occupies most of its short connector. | `b_e_graph_sum` | P2 | acknowledged |
| RT-06 | Typography | Input edge label X overlays the arrow shaft, though it remains readable. | `a_e_input_split` | P2 | acknowledged |
| RT-07 | Arrow hygiene | Recent-conditioning route has three orthogonal segments and is visually more complex than other auxiliary edges. | `a_e_recent_condition` | P2 | acknowledged |
| RT-08 | Box integrity | RGSP and DHC overview cards have more padding than the role cards. | `a_rgsp`, `a_dhc` | P2 | acknowledged |
| RT-09 | Layout | Pale-blue page background remains visible between the white lower panels. | graph background | P2 | acknowledged |
| RT-10 | Typography | Trend-output tensor is near the lower route endpoint rather than centered on the full route. | `b_e_trend_sum` | P2 | acknowledged |
| RT-11 | Spacing | Patch-to-Bi-Mamba gap is larger than Bi-Mamba-to-variable-graph gap. | `b_patches`, `b_bimamba`, `b_graph` | P2 | acknowledged |
| RT-12 | Box integrity | Shared decoder remains the tallest operator and therefore has more internal padding. | `c_decoder` | P2 | acknowledged |
| RT-13 | Typography | Paired-output label uses compact symbols rather than the full `y^{+m}, y^{-m}` notation. | `c_e_decoder_diff` | P2 | intentional abbreviation |
| RT-14 | Icons | Schematic sine waves do not reproduce the irregular reference traces. | waveform cells | P2 | intentional editable approximation |
| RT-15 | Style coherence | Detail panels are slightly more spacious than the denser KARMA reference subpanels. | panels b/c | P2 | intentional clarity trade-off |
| RT-16 | Text readability | Secondary tensor labels are the smallest text class and approach the lower paper-scale limit. | all edge labels | P2 | retained; still legible in full-width manuscript render |
| RT-17 | Text readability | Superscripts are visually lighter than module labels after PDF scaling. | tensor labels | P2 | retained as conventional notation |
| RT-18 | Arrow hygiene | The lower RGSP trend route is long because it must reach the right-side sum without crossing the seasonal path. | `b_e_trend_sum` | P2 | retained for semantic separation |
| RT-19 | Arrow hygiene | The two diagonal arrows entering the overview sum differ from the otherwise orthogonal grammar. | `a_e_base_sum`, `a_e_delta_sum` | P2 | retained to make fan-in immediately recognizable |
| RT-20 | Box integrity | The ΔY overview card is larger than its short glyph requires. | `a_delta` | P2 | retained to match the Base output card |
| RT-21 | Box integrity | Diff. is wider than its four-character label requires. | `c_diff` | P2 | retained for balanced operator scale |
| RT-22 | Spacing | The vertical gap between overview branches is larger than the gap between their output cards and the sum. | panel a | P2 | retained to separate roles clearly |
| RT-23 | Spacing | RGSP lower route leaves an intentionally large white region around its long projection line. | panel b | P2 | retained to avoid crossings and small boxes |
| RT-24 | Color and palette | Green is reused for decomposition, graph propagation, pooling, and recent-state extraction. | green transform blocks | P2 | retained as one shared transform category |
| RT-25 | Color and palette | Output teal is used for Base, ΔY, and Gate rather than a separate color per output role. | teal blocks | P2 | retained to avoid palette scatter |
| RT-26 | Typography | Acronym lines are bold while explanatory second lines are regular in RGSP/DHC cards. | `a_rgsp`, `a_dhc` | P2 | intentional hierarchy |
| RT-27 | Layout | The overview panel is shorter than the detail panels, creating a 40/60 vertical split rather than equal bands. | panel geometry | P2 | retained because details need more height |
| RT-28 | Icons | Input/output waves illustrate three variables but do not show axes or sample counts. | waveform cells | P2 | intentional schematic icon, not a plot |
| RT-29 | Style coherence | No legend is embedded for blue/orange branches. | full figure | P2 | direct Recent/Distant labels make a legend redundant |
| RT-30 | Style coherence | No caption is embedded inside the canvas. | full figure | P2 | caption remains in LaTeX per journal convention |

The red-team scan found no missing entity, wrong direction, box overlap, clipped text, or arrow-through-box defect. P1 items RT-01–RT-04 are fixed in the final patch; P2 items are documented as low-risk presentation choices.

## Requirement And Semantic Audit

| check | observed screenshot | expected from source | actual | status |
|---|---|---|---|---|
| Overall role split | `review/final-redteam.png`, panel a | input splits into recent and distant histories | explicit blue/orange fan-out | pass |
| Base path | panel a/b | RGSP alone generates full base forecast | RGSP ends at Base Ŷb; b ends at Ŷb | pass |
| Correction path | panel a/c | DHC generates ΔY only | DHC ends at correction card and Gate→ΔY | pass |
| DHC conditioning | panel a/c | recent reference conditions distant correction | dashed recent-ref. edge plus separate recent-state decoder entry | pass |
| RGSP internals | panel b | decomposition, patches, Bi-Mamba, variable graph, trend projection | all present in forward order | pass |
| DHC internals | panel c | AvgPool, recent state, shared decoder, difference, gate | all present with separate upper/lower decoder inputs | pass |
| Vertical reading | panels a/b/c | long operator labels read top-to-bottom | De-norm, Decomp., Bi-Mamba, Variable graph, AvgPool, Recent state, Shared decoder all comply | pass |
| Terminal placement | panels b/c | outputs at the right, not below | Ŷb and ΔY are right-aligned terminals | pass |

## Red-Team Fix Verification

| check | observed screenshot | finding | XML cells to change | status |
|---|---|---|---|---|
| RT-01 | `cycle4-final.png` → `final-redteam.png` | output waveform/De-norm centers now coincide | `a_output_wave_*` | fixed |
| RT-02 | `cycle4-final.png` → `final-redteam.png` | decoder, Diff., Gate, and ΔY now share one centerline | `c_diff`, `c_gate`, `c_output` | fixed |
| RT-03 | `cycle4-final.png` → `final-redteam.png` | DHC output gains visible right margin | `c_gate`, `c_output` | fixed |
| RT-04 | `cycle4-final.png` → `final-redteam.png` | RGSP output moved left with a readable final arrow | `b_output` | fixed |

## User-Found Defects — 2026-08-31

The prior 44/50 self-score is withdrawn. The editor-scale screenshot exposed three P1 defects that the earlier audit did not weight strongly enough.

| id | severity | user-visible defect | corrective change | verification |
|---|---|---|---|---|
| UF-01 | P1 | Panel b was cramped: small boxes and short gaps reduced paper-scale readability. | Widened panel b from 930 px to 1070 px and redistributed the upper chain onto a 650 px centerline with longer arrow spans. | `review/user-revision1.png`; fixed |
| UF-02 | P1 | Vertical labels in panel c read bottom-to-top. | Applied `horizontal=0;rotation=180` to reverse Draw.io's default −90° vertical text rotation; the same top-to-bottom rule is used for every vertical operator. | `review/user-mainflow-final.png`; fixed |
| UF-03 | P1 | Panel c connectors used visible bends despite a straight-line visual grammar. | Aligned both input rows and the decoder/Diff./Gate/output row; removed orthogonal routing where a direct horizontal connector is intended. | `review/user-revision1.png`; fixed |
| UF-04 | P1 | The DHC output label sat too close to the right boundary after the re-layout. | Rebalanced Diff., Gate, and ΔY positions while preserving the 720 px output centerline. | `review/user-revision2.png`; fixed |

## Focused Re-Red-Team Audit

| id | zone | finding | severity | status |
|---|---|---|---|---|
| URT-01 | panel b | Upper RGSP chain now has distinct, readable arrow spans between all operators. | P1 | fixed |
| URT-02 | panel b | Seasonal and trend branches remain visually separated with no crossing. | P1 | fixed |
| URT-03 | panel b | Trend route is long but entirely orthogonal and semantically unambiguous. | P2 | retained |
| URT-04 | panel b | Variable graph, sum, and base output no longer collide with the panel boundary. | P1 | fixed |
| URT-05 | panel b | Vertical operator labels read from top to bottom. | P1 | fixed |
| URT-06 | panel c | Distant-to-AvgPool and recent-to-state connectors are horizontal. | P1 | fixed |
| URT-07 | panel c | Pool/state-to-decoder connectors are horizontal and enter separate decoder ports. | P1 | fixed |
| URT-08 | panel c | Decoder-to-Diff.-to-Gate-to-ΔY lies on one centerline. | P1 | fixed |
| URT-09 | panel c | AvgPool, Recent state, and Shared decoder read from top to bottom. | P1 | fixed |
| URT-10 | panel c | Final ΔY has visible right padding after output-lane rebalance. | P1 | fixed |
| URT-11 | lower panels | Module font sizes remain 24–30 px and dominate 20–22 px tensor annotations. | P2 | accepted |
| URT-12 | lower panels | Dashed panel borders do not intersect any module or terminal glyph. | P1 | fixed |
| URT-13 | full figure | No curved connector remains; all paths are straight or orthogonal. | P1 | fixed |
| URT-14 | full figure | Pale-blue gutter remains a deliberate grouping field around the lower white cards. | P2 | accepted |
| URT-15 | paper scale | Final decision is based on the compiled manuscript page, not the editor screenshot alone. | P1 | verified in final page render |

## Self-Score (revised after user audit)

| dimension | evidence | score |
|---|---|---:|
| Text readability | Main labels are readable, but tensor annotations remain the smallest class and vertical words require careful paper-scale review. | 8/10 |
| Arrow accuracy | All directions, fan-outs, fan-ins, and DHC decoder entries are correct; one point deducted for the necessarily long RGSP trend route. | 9/10 |
| Color coherence | Six muted semantic fills plus two role accents are consistent; one point deducted for pale-blue gutters outside the lower white panels. | 9/10 |
| Layout consistency | The revised b/c geometry is aligned, but the long trend route and unequal panel widths remain visible trade-offs. | 7/10 |
| Style match to reference/spec | Large boxes and top-to-bottom operators follow the reference family, while the schematic waveforms and lower density remain simplified. | 8/10 |
| **TOTAL** | ALLOWED: total ≥40 and every dimension ≥6. | **41/50** |

## Remaining Gaps

| gap | severity | reason | next action |
|---|---|---|---|
| Sine waves are smoother than the reference traces | P2 | kept as fully editable Draw.io primitives | replace only if the user supplies exact editable traces |
| Pale-blue gutter remains between white detail panels | P2 | page background creates the overview field without adding a validator-conflicting background rectangle | optional cosmetic adjustment in Draw.io |

## User Main-Flow Replacement — 2026-08-31

The user-authored `paper/neurocomputing/主流程图.drawio` replaced the prior canonical source. The visual composition was retained. Three detached connectors were rebound to their intended nodes, panel-local C-section coordinates were flattened to stable page coordinates, and the small input indicator was routed left of the waveforms to remove a curve crossing. Static validation reports zero FAIL items, and the replacement was verified in the compiled manuscript page.

## Panel Label And Expanded-Name Revision — 2026-08-31

Panel labels `b` and `c` were separated from their headings and placed at the lower-left corners to match panel `a`. The manuscript-defined expansions, `Recent Graph-Enhanced State-Space Predictor` and `Distant-History Corrector`, are centered at the top of panels b and c in title case. The canvas preview shows no title clipping or overlap.

## Title Clearance And Initial Emphasis — 2026-08-31

All internal modules and their explicit waypoints in panels b and c were shifted downward by 25 px while the containers, titles, and lower-left panel labels remained fixed. Only the first letter of each title word is bold (`R/G/E/S/S/P` and `D/H/C`); the remaining letters use regular weight. The full-canvas export confirms clear vertical separation between headings and modules without bottom-border collisions.

## Role-Aware Waveform Revision — 2026-08-31

The overview waveform icons now communicate the model's function instead of serving as generic decoration. The input is split into a dashed orange distant segment (`X^d`) and a solid blue recent segment (`X^r`) with a visible gap. The output is shown as a coherent green final forecast, with the compact colored relation `Ŷ^b + ΔY → Ŷ` directly above it. This preserves the three-variable schematic while exposing role separation and correction-based prediction without adding a legend or ground-truth curve.

| check | evidence | result |
|---|---|---|
| input role distinction | `review/waveform-role-effect-revision.png` | orange distant and blue recent segments are visually separable |
| output mechanism | same preview | base, correction, and final symbols use the same blue/orange/green grammar as the architecture |
| frame containment | full-canvas preview | all waveform strokes and labels stay within panel a |
| paper-scale legibility | `review/manuscript-page4-waveform-role-effect.png` | role tags and correction equation remain readable in the compiled two-column page |
| static quality gate | `validate_drawio.py`, `validate_visual_quality.py` | 0 FAIL |

## Submission-Polish Revision — 2026-08-31

The figure was tightened against the SciPilot publication checklist while retaining the established Draw.io architecture. The overview role tags and the output relation were enlarged so that normal symbols render above 7 pt and the smallest superscript renders above 6 pt at manuscript width. Distant-history waveform strokes were raised to full opacity and 3 px while remaining dashed, preserving role distinction in grayscale. Panel labels and manuscript references were synchronized to `(A)`, `(B)`, and `(C)`, with equal lower-left insets and aligned lower-panel geometry.

Font export was tested rather than inferred from the Draw.io style name. `Arial` fell back to embedded DejaVu Sans in Chromium; explicit Nimbus Sans produced Type 3 fonts and was rejected. The final source therefore uses explicit DejaVu Sans, and the PDF embeds DejaVu Sans as CID TrueType with Unicode mapping. The raster derivatives now carry 300 DPI PNG metadata and a true 7695×3915, 1000 DPI LZW TIFF export; the vector PDF remains the preferred submission artifact.

| check | evidence | result |
|---|---|---|
| paper-scale minimum text | PDF text bounding boxes | output equation ≈7.6 pt; superscript ≈6.3 pt |
| color/grayscale redundancy | `review/manuscript-page4-submission-polish-gray.png` | dashed distant and solid recent roles remain distinguishable |
| panel labels | `review/manuscript-page4-submission-polish.png` | `(A)/(B)/(C)` consistent in figure, caption, and body references |
| font embedding | `pdffonts figures/Fig1_TimeRole_Architecture.pdf` | CID TrueType, embedded/subset, no Type 3 |
| file compliance | PNG/TIFF metadata audit | PNG 300 DPI; TIFF 1000 DPI |
| structural validation | Draw.io validators | 0 FAIL |

## Normalized-Output Symbol Revision — 2026-09-01

To make the overview's post-sum De-norm operation explicit, the two operands immediately before the summation node were relabeled from `Ŷ^b` and `ΔY` to `Ŷ^b_norm` and `ΔY_norm`. Geometry, connectors, styles, and the detailed module panels were left unchanged. The canonical Draw.io source and the user working copy `主流程图.drawio` were synchronized.

| check | evidence | result |
|---|---|---|
| static quality gate | `validate_drawio.py`, `validate_visual_quality.py` | 0 FAIL; 12 pre-existing conservative warnings reviewed |
| direct export | `figures/Fig1_TimeRole_Architecture.pdf` | both `norm` subscripts render inside their boxes |
| paper-scale review | compiled `main.pdf`, page 4 | both normalized operands remain readable at manuscript width |
| font embedding | `pdffonts figures/Fig1_TimeRole_Architecture.pdf` | embedded/subset CID TrueType, no Type 3 |

## Waveform Enrichment Review — 2026-09-01

### Pass 1 — full-canvas preview

| zone | finding | severity | correction |
|---|---|---|---|
| top-left | four input traces are visible and remain within panel A | P2 | retained |
| top-left | input-to-Role-split connector contains a small vertical jog | P1 | replace the large hidden anchor with a 2×2 aligned connection point |
| top-left | `L` and `D` dimension markers are legible | P2 | retained |
| top-left | role boundary separates `X^d` and `X^r` without cutting labels | P2 | retained |
| top-center | new waveform content does not shift the architecture modules | P1 | verified |
| top-right | output connector contains a small vertical jog | P1 | align output anchor with the De-norm centerline |
| top-right | output `D` marker remains within the dashed container | P2 | retained |
| top-right | `T` arrow is separated from the Forecast label | P2 | retained |

### Pass 2 — corrected full-canvas preview

| zone | finding | severity | status |
|---|---|---|---|
| top-left | input arrow is horizontal from the trace bundle to Role split | P1 | fixed |
| top-left | four channel traces have distinct shapes and colors | P1 | fixed |
| top-left | `X^d/X^r` separator remains visible at full-canvas scale | P2 | verified |
| top-right | De-norm-to-forecast arrow is horizontal | P1 | fixed |
| top-right | output traces and `T,D` markers have no clipping | P1 | verified |
| center | the base/correction fan-in is unchanged | P1 | verified |
| bottom-left | panel B is unchanged | P1 | verified |
| bottom-right | panel C is unchanged | P1 | verified |

### Pass 3 — compiled manuscript review

| zone | finding | severity | status |
|---|---|---|---|
| figure top-left | four input traces remain distinguishable at manuscript width | P1 | verified |
| figure top-left | `X∈R^(L×D)`, `L`, and `D` remain readable | P1 | verified |
| figure top-left | `X^d/X^r` remain subordinate to the TimeRole title | P2 | verified |
| figure top-right | four forecast traces remain distinguishable | P1 | verified |
| figure top-right | `Ŷ∈R^(T×D)`, `T`, and `D` remain readable | P1 | verified |
| figure center | new waveform density does not compete with RGSP/DHC | P1 | verified |
| figure bottom | B/C titles and internal modules are unaffected | P1 | verified |
| page | figure, caption, and following two-column text retain the prior layout | P1 | verified |

### Screenshot Evidence

| pass | screenshot path | capture type | full canvas visible | notes |
|---|---|---|---|---|
| 1 | `review/waveform-enrichment-cycle1.png` | canvas-only | yes | initial enriched waveform layout |
| 2 | `review/waveform-enrichment-cycle2.png` | canvas-only | yes | straightened input/output connectors |
| 3 | `review/manuscript-page4-waveform-enrichment.png` | full-page | yes | final paper-scale check |

### Focused Red-Team Audit

| check | finding | status |
|---|---|---|
| arrow direction | input enters Role split; De-norm points to forecast | pass |
| arrowhead placement | both waveform connectors terminate outside the traces/modules | pass |
| trace containment | no trace crosses the panel border | pass |
| role semantics | dashed separator and labels encode the distant/recent partition | pass |
| tensor semantics | `L,D,T` correspond to real input/output axes | pass |
| color semantics | four colors denote representative variables, not extra modules | pass |
| label hierarchy | tensor dimensions remain below module and panel titles | pass |
| connector crossing | no new connector crosses text or another module | pass |
| output clarity | forecast bundle is distinct from the green Gate block in panel C | pass |
| grayscale redundancy | four non-identical shapes remain distinguishable without color | pass |

### Self-Score

| dimension | evidence | score |
|---|---|---:|
| Text readability | tensor axes and role labels survive manuscript-width rendering | 9/10 |
| Arrow accuracy | both external connectors are straight and correctly directed | 10/10 |
| Color coherence | only the established four semantic colors are reused | 9/10 |
| Layout consistency | waveform bundles fill their regions without moving modules | 9/10 |
| Style match | multichannel traces and tensor axes follow the reference visual grammar | 9/10 |
| **TOTAL** | all dimensions ≥6 and total ≥40 | **46/50** |

### Remaining Gaps

| gap | severity | reason |
|---|---|---|
| traces are schematic rather than real samples | P2 | a framework diagram should not imply a selected experimental case |
| static geometry validator reports legacy panel-parent false positives | P2 | its collision pass does not resolve panel-local coordinates; direct PDF and manuscript screenshots are clean |

## Waveform Enrichment Plan — 2026-09-01

| issue | reference evidence | planned fix |
|---|---|---|
| input/output traces are overly regular | reference uses irregular multichannel traces | replace sine primitives with four editable waypoint polylines per side |
| tensor geometry is implicit | reference marks temporal and variable axes | add `L,D` at input and `T,D` at output |
| role partition and multivariate identity compete | current glyph uses only two role colors | use channel-colored curves plus a dashed `X^d/X^r` temporal boundary |
| output waveform has limited density | three identical green waves leave unused area | use four distinct forecast traces while retaining the green final-output label |

This is a local panel-A revision. The overall architecture and panels B/C are frozen.

## Panel-B Normalized-Input Symbol Revision — 2026-09-01

Panel B's input label was changed from `X^r` to `X̃^r` so that the diagram matches the normalized recent input consumed by RGSP. No geometry, connector, font, color, module, or other tensor label was changed. The user working source `主框架.drawio` and the canonical figure source remain byte-identical.

| check | evidence | result |
|---|---|---|
| target-cell audit | Draw.io cell `b_input` | exactly one label changed to `\widetilde{X}^{r}` in each synchronized source |
| structural validation | `validate_drawio.py` | XML valid; no duplicate IDs, external images, or placeholder labels |
| heuristic visual validation | `validate_visual_quality.py` | reports pre-existing false positives caused by panel-local coordinates and MathJax sizing; no new geometry was introduced |
| direct export | `figures/Fig1_TimeRole_Architecture.pdf` | tilde, superscript, and box boundaries render correctly |
| paper-scale review | compiled `main.pdf`, page 4 | `X̃^r` remains legible and contained at manuscript width |
| font embedding | `pdffonts figures/Fig1_TimeRole_Architecture.pdf` | embedded/subset CID TrueType, no Type 3 |
