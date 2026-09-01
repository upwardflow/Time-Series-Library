# Layout Grid

## Canvas

- width: 2000
- height: 1040
- scale assumption: exported full canvas and placed at manuscript `\textwidth`
- margin: 40–50 px

## Grid Lines

| name | x | y | purpose |
|---|---:|---:|---|
| overview-top | 40 | 25 | panel a boundary |
| overview-upper-flow | 0 | 153 | recent branch center |
| overview-lower-flow | 0 | 329 | distant branch center |
| detail-top | 0 | 480 | panels b/c boundary |
| detail-flow-upper | 0 | 650 | upper detail paths |
| detail-flow-lower | 0 | 850 | lower detail paths |
| split-detail | 1120 | 0 | center gutter between b/c |

## Region Boxes

| id | x | y | w | h |
|---|---:|---:|---:|---:|
| panel_a | 40 | 25 | 1920 | 420 |
| panel_b | 40 | 480 | 1070 | 520 |
| panel_c | 1130 | 480 | 830 | 520 |

## Repeated Components

| family | count | cell size | spacing | start x,y |
|---|---:|---|---|---|
| overview role cards | 2 | 180×82 | 94 vertical | 485,112 |
| overview major modules | 2 | 250×122 | 54 vertical | 770,92 |
| overview output cards | 2 | 190×82 | 94 vertical | 1130,112 |
| DHC input rows | 2 | 100×80 | 115 vertical | 1160,614 |
| DHC state blocks | 2 | 60×170 | 25 vertical | 1320,568 |
| input wave segments | 6 | 55×24 | 10 horizontal role gap, 6 vertical | 75,173 |
| output wave lines | 3 | 120×24 | 6 vertical | 1760,178 |

## Waveform Enrichment Grid — 2026-09-01

| family | count | span | spacing | region |
|---|---:|---:|---:|---|
| input channel traces | 4 | x=60–190 | 28 px vertical | y=158–242 |
| output channel traces | 4 | x=1735–1915 | 28 px vertical | y=158–242 |
| input role separator | 1 | x=140 | y=145–251 | input bundle |
| input dimension arrows | 2 | L: x=60–190; D: y=148–250 | — | input bundle |
| output dimension arrows | 2 | T: x=1735–1915; D: y=148–250 | — | output bundle |

The transparent input/output anchor vertices retain connector attachment without adding visible boxes. All new traces are page-level editable edges and stay inside panel A.

## Major Component Coordinates

| id | x | y | w | h |
|---|---:|---:|---:|---:|
| input_anchor | 75 | 131 | 120 | 131 |
| role_split | 245 | 160 | 170 | 115 |
| recent | 485 | 112 | 180 | 82 |
| distant | 485 | 288 | 180 | 82 |
| rgsp | 770 | 92 | 250 | 122 |
| dhc | 770 | 268 | 250 | 122 |
| base | 1130 | 112 | 190 | 82 |
| correction | 1130 | 288 | 190 | 82 |
| sum | 1405 | 183 | 76 | 76 |
| denorm | 1550 | 145 | 95 | 150 |
| output_anchor | 1700 | 122 | 230 | 140 |
| b_input | 75 | 635 | 110 | 80 |
| b_decomp | 250 | 570 | 100 | 210 |
| b_patches | 430 | 625 | 190 | 100 |
| b_bimamba | 700 | 580 | 95 | 190 |
| b_graph | 850 | 580 | 95 | 190 |
| b_sum | 980 | 650 | 50 | 50 |
| b_trend | 480 | 825 | 100 | 150 |
| c_distant | 1160 | 614 | 100 | 80 |
| c_recent | 1160 | 809 | 100 | 80 |
| c_pool | 1320 | 568 | 60 | 170 |
| c_state | 1320 | 763 | 60 | 170 |
| c_decoder | 1494 | 551 | 67.5 | 395 |
| c_diff | 1640 | 706 | 105 | 90 |
| c_gate | 1784 | 709 | 85 | 80 |
| c_output | 1900 | 723 | 50 | 50 |

## Drawing Order

1. page background and panel containers
2. module boxes and waveform primitives
3. connectors with fixed ports/waypoints
4. panel titles and output labels
5. edge tensor labels
