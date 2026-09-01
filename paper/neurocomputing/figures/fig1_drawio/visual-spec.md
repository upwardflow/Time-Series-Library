# Visual Spec

## Source

- Reference image: user-provided KARMA architecture screenshot in the conversation (style/layout only).
- Target drawio: `Fig1_TimeRole_Architecture.drawio`.
- Canvas: 2000 × 1040, landscape, content-filled crop.
- Font policy: explicit DejaVu Sans throughout; headings 31–38 px, modules 24–30 px, tensor annotations 20–25 px. This export-safe choice preserves embedded CID TrueType fonts in the current Linux draw.io pipeline.

## Global Style

- Background: white page with a pale-blue overall panel and white detail panels.
- Primary font: DejaVu Sans; verify CID TrueType embedding after each final export. Nimbus Sans was rejected because Chromium exported it as Type 3.
- Stroke style: 2.5–3 px solid module borders; 3 px dashed semantic containers.
- Arrow style: 3 px filled classic arrows, straight or orthogonal.
- Color palette: muted blue/green/purple/orange with charcoal neutral flow.

## Style Extraction: KARMA Architecture Screenshot

### 1. Palette

| role | hex | used on |
|---|---|---|
| background | `#FFFFFF` | page |
| overview background | `#EAF3F5` | panel a |
| primary fill | `#DDEBF4` | recent/RGSP blocks |
| secondary fill | `#E4EEDC` | transforms/graph/state blocks |
| accent / highlight | `#F8E2D2` | distant/DHC/trend blocks |
| transform fill | `#E9E2F1` | split/decoder/de-normalization |
| output fill | `#E1F0F0` | prediction cards/gate |
| border stroke | `#9EA8AD` | dashed containers |
| arrow/body text | `#30383D` | primary flow and text |
| recent accent | `#4E8DBB` | recent branch and outlines |
| distant accent | `#D98243` | distant branch and outlines |

Total distinct semantic fills: 6; branch accents: 2.

### 2. Typography

- Heading font: DejaVu Sans, 38 px, bold.
- Panel heading font: DejaVu Sans, 31–32 px, bold.
- Module text font: DejaVu Sans, 24–30 px, semibold/bold.
- Small tensor label: DejaVu Sans, 20–25 px.
- Code/mono font: none.

### 3. Shape Language

- Corner radius: subtle rounded rectangles, approximately 14–18 px.
- Stroke width for boxes: 2.5–3 px.
- Stroke width for arrows: 3 px.
- Dash pattern for containers: `10 8`.
- Shadow: no.
- Fill opacity for background regions: 100% pale tint.

### 4. Layout Rhythm

- Outer margin: 40–55 px.
- Gap between major regions: 35 px.
- Same-row horizontal gap: 55–100 px.
- Internal padding: 14–20 px vertically, 18–24 px horizontally.
- Typical horizontal module: 150–240 × 76–112 px.
- Typical vertical operator: 80–105 × 140–205 px.
- Grid alignment: 5/10 px.

### 5. Arrow Grammar

- Default arrow: filled classic triangular head.
- Arrow color: charcoal for neutral flow; blue/orange for role-specific flow.
- Arrowhead size: medium.
- Routing: straight or orthogonal; no curved connectors.
- Labels: only tensor/relationship labels, 20–22 px.
- Color coding: yes, only recent vs distant roles.

### 6. Icon Language

- Minimal editable primitives.
- Three small sine-wave rows represent multivariate sequences; the input is split into dashed orange distant history and solid blue recent history.
- The output uses solid green forecast curves and a compact colored equation (`base + correction → final`) to expose the model effect without an extra legend.
- No external icons, logos, or raster imagery.

### 7. Density & Composition

- Diagram type: multi-panel architecture.
- Major regions: 3.
- Density: medium-dense.
- Whitespace: moderate and even; panels should feel filled rather than hollow.
- Panel labels: Elsevier-style `(A)`, `(B)`, `(C)`, consistently placed at the lower-left with equal bottom inset.
- Legend: omitted because the role branches are directly named and consistently colored.
- Caption: provided by LaTeX, not embedded in figure.

## Semantic Justification

| element | visual form | method meaning | each unit corresponds to | justified? |
|---|---|---|---|---|
| input waveform | three pairs of editable sine segments | distant/recent role split in the observed history | orange dashed segment = distant; blue solid segment = recent | yes |
| output waveform | three editable green sine lines plus compact equation | corrected multivariate forecast | `Ŷᵇ + ΔY → Ŷ` states the correction mechanism | yes |
| blue row | rounded modules and blue arrows | recent high-resolution base path | one block = one named stage | yes |
| orange row | rounded modules and orange arrows | distant compressed correction path | one block = one named stage | yes |
| dashed blue connector | auxiliary edge | recent reference conditions DHC | one edge = conditioning relation | yes |
| vertical blocks | narrow rounded operators | compact operators with long names | one block = one operator | yes |
| dashed panel borders | group containers | overall/RGSP/DHC semantic groups | one border = one module view | yes |
| decorative token bars/grids | omitted | no corresponding method entity | n/a | no—deleted |

## Regions

| id | bbox x,y,w,h | role | visual notes |
|---|---|---|---|
| panel_a | 40,25,1920,420 | overall architecture | pale blue rounded background |
| panel_b | 40,480,1070,520 | RGSP detail | widened white dashed container; long arrows separate the two branches |
| panel_c | 1130,480,830,520 | DHC detail | compact white dashed container with two aligned input rows |

## Text Blocks

| id | bbox x,y,w,h | text | font | alignment | priority |
|---|---|---|---|---|---|
| title | 75,45,300,48 | TimeRole | 40 bold | left | highest |
| panel_b_title | 150,492,850,48 | Recent Graph-Enhanced State-Space Predictor | 31; initials bold | centered above panel b | highest |
| panel_c_title | 1200,492,690,48 | Distant-History Corrector | 31; initials bold | centered above panel c | highest |
| panel_b_label | 65,936,75,50 | (B) | 32 bold | lower left, matching panel A | highest |
| panel_c_label | 1155,936,75,50 | (C) | 32 bold | lower left, matching panel A | highest |
| tensor labels | adjacent to edges | exact compact symbols | 20–22 | centered | secondary |

## Shapes

| id | bbox x,y,w,h | type | fill | stroke | notes |
|---|---|---|---|---|---|
| role_split | 245,160,170,115 | rounded rect | purple | purple | fan-out |
| recent / distant | 485,112/288,180,82 | rounded rect | blue/orange | role accent | two roles |
| rgsp / dhc | 770,92/268,250,122 | rounded rect | blue/orange | role accent | major stages |
| base / correction | 1130,112/288,190,82 | rounded rect | output | role accent | same output shape |
| sum | 1405,183,76,76 | ellipse | white | charcoal | prediction-space addition |
| denorm | 1550,145,95,150 | vertical rounded rect | purple | purple | top-to-bottom label |

## Connectors

| id | from | to | route | arrowheads | label | notes |
|---|---|---|---|---|---|---|
| a1 | input | role_split | straight | end | X | main flow |
| a2/a3 | role_split | recent/distant | orthogonal | end | Xr/Xd | role fan-out |
| a4/a5 | role nodes | RGSP/DHC | straight | end | none | role paths |
| a6 | recent | DHC | orthogonal dashed | end | recent ref. | conditioning |
| a7/a8 | RGSP/DHC | base/correction | straight | end | Yb/ΔY | prediction-space outputs |
| a9/a10 | base/correction | sum | orthogonal | end | none | fan-in |
| a11/a12 | sum | denorm | straight | end | none | final path |

## Semantic Relations And Flow

| id | source | target | meaning | direction/cardinality | visual evidence |
|---|---|---|---|---|---|
| SR1 | recent history | RGSP | complete base prediction | one-to-one | upper blue row |
| SR2 | distant history + recent reference | DHC | conditional correction | two-to-one | orange row plus dashed blue condition |
| SR3 | base + correction | final normalized forecast | fan-in | sum ellipse |
| SR4 | normalized forecast | forecast | one-to-one | De-norm block |

## Icons And Images

| id | bbox x,y,w,h | meaning | exact/approx/missing | replacement plan |
|---|---|---|---|---|
| wave_input | 75,131,120,131 | role-differentiated multivariate input | editable approximation | six sine-wave segments, enlarged role tags, and a visible role gap |
| wave_output | 1700,122,230,140 | corrected multivariate forecast | editable approximation | three green sine-wave primitives plus enlarged colored correction equation |

## Waveform Enrichment Amendment — 2026-09-01

### Reference-derived style contract

| parameter | reference observation | selected treatment |
|---|---|---|
| waveform language | four compact, irregular multivariate traces | four editable piecewise-linear traces at each end |
| colors | muted red, green, blue, and purple/gray | reuse the established orange, green, blue, and purple TimeRole palette |
| dimensions | horizontal length and vertical variable-count arrows | input: `L` and `D`; output: `T` and `D` |
| typography | small tensor dimensions next to the waveform | 18–20 px, subordinate to module labels |
| density | compact waveform bundles with little unused space | fill the existing input/output regions without moving the architecture |
| arrow grammar | dark straight arrows between waveform and model | preserve the current straight, filled-arrow main path |

### Updated semantic justification

| element | visual form | method meaning | each unit corresponds to | justified? |
|---|---|---|---|---|
| input trace bundle | four colored piecewise-linear curves | representative channels of the multivariate history | one curve = one representative variable | yes |
| role boundary | thin dashed vertical separator with `X^d`/`X^r` | distant/recent temporal partition subsequently performed by Role split | left/right interval = one temporal role | yes |
| input dimension arrows | horizontal `L`, vertical `D` | history length and number of variables | one arrow = one tensor axis | yes |
| output trace bundle | four colored piecewise-linear curves | representative channels of the multivariate forecast | one curve = the corresponding forecast variable | yes |
| output dimension arrows | horizontal `T`, vertical `D` | prediction length and number of variables | one arrow = one tensor axis | yes |

The waveform values remain schematic and are not experimental observations. Their role is to expose the multivariate tensor geometry and the distant/recent partition while retaining full Draw.io editability.
