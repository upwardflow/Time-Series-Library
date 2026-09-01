# Diagram Brief

## User Goal

- Output: restart Figure 1 as an editable Draw.io research figure and export PDF/SVG/PNG for the manuscript.
- Audience: Neurocomputing reviewers and readers of multivariate long-term forecasting papers.
- Must communicate: input history is split by temporal role; RGSP produces a complete base forecast from the recent window; DHC produces only a distant-history correction; the two outputs are added in prediction space.
- Must not do: reproduce the prior small, crowded lower panels; place waveforms outside containers; use curved/bent decorative connectors; overload panels b/c with implementation details; rotate vertical text bottom-to-top.

## Source Inventory

| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | `manuscript_zh.tex`, Sec. 3 | paper text | content + structure | must | authoritative module names, variables, and directions |
| S2 | current Figure 1 and its generator | local figure/code | content cross-check | should | retained as backup; not used as the new visual source |
| S3 | user-provided KARMA architecture screenshot | screenshot | style + layout | must | large pale overview panel, two dashed detail panels, readable boxes, narrow vertical operators |
| S4 | iterative user feedback in this thread | feedback | visual constraints | must | larger fonts/boxes, longer arrows, less empty space, top-to-bottom vertical labels |

## Requirement Traceability

| id | requirement | source evidence | priority | planned visual encoding |
|---|---|---|---|---|
| R1 | show role-differentiated recent/distant history | S1 | must | explicit Role split with blue recent and orange distant branches |
| R2 | RGSP outputs the full base forecast | S1 | must | upper branch terminates at Base forecast before sum |
| R3 | DHC outputs a correction only | S1 | must | lower branch terminates at correction ΔY before sum |
| R4 | show recent conditioning of DHC | S1 | must | short dashed blue connector labeled recent ref. |
| R5 | b shows decomposition, dual-scale patching, shared Bi-Mamba, variable graph, and trend projection | S1 | must | compact two-path RGSP detail panel |
| R6 | c shows distant compression, recent state, shared decoder, paired difference, and gate | S1 | must | two-row DHC detail panel with clean fan-in |
| R7 | typography and boxes must resemble the reference scale | S3/S4 | must | 26–30 px module labels, 34–40 px headings, substantial vertical blocks |
| R8 | long operator names read top-to-bottom | S4 | must | `horizontal=0;direction=south` vertical text |
| R9 | b/c arrows visibly span the gaps | S4 | must | 55–100 px inter-module gaps and orthogonal routes |
| R10 | editable source and publication exports | user request + skill | must | `.drawio` source of truth; PDF/SVG/PNG derived exports |

## Semantic Model

| id | entity or relationship | direction / hierarchy / cardinality | visual encoding | uncertainty |
|---|---|---|---|---|
| M1 | Input → Role split | one-to-one | dark main-flow arrow | none |
| M2 | Role split → Recent / Distant | one-to-two | blue/orange branch arrows | none |
| M3 | Recent → RGSP → base forecast | serial | blue row | none |
| M4 | Distant → DHC → correction | serial | orange row | none |
| M5 | Recent → DHC | auxiliary conditioning | dashed blue arrow | none |
| M6 | base forecast + correction → forecast | fan-in then serial | sum node, De-norm, output | none |
| M7 | RGSP seasonal/trend paths | one-to-two then fan-in | upper blue/green path and lower orange path | none |
| M8 | DHC distant/recent states | two-to-one | aligned rows entering shared decoder | none |
| M9 | paired outputs → difference → gate | serial | white Diff. then green Gate | none |

## Style Contract

| id | font | palette | stroke | icon style | layout density | reference source |
|---|---|---|---|---|---|---|
| C1 | Helvetica/Arial | pale blue, muted blue/green/purple/orange, charcoal | 2.5–3 px; dashed groups | editable wave primitives only | medium-dense, low reading burden | KARMA screenshot |

## Open Assumptions

| assumption | risk | how to verify |
|---|---|---|
| `recent ref.` is sufficient for the top auxiliary edge | low | compare against method equations and caption |
| paired decoder outputs can be summarized as `y+ / y−` on one edge | low | keep Diff. and exact ΔY output explicit |
| no separate legend is needed because branch colors are repeated and directly named | low | inspect paper-scale screenshot for ambiguity |
