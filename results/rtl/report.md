# RTL — valid-gated register updates (independent of the GPU experiments)

Scoreboard across all runs: **PASS** (3100 transactions, 0 mismatches).

## Headline configuration: 25% valid duty, new random inputs every cycle

| scope            | metric      |   baseline |   optimized |   reduction_pct |   valid_duty_pct | inputs_held_when_invalid   |
|:-----------------|:------------|-----------:|------------:|----------------:|-----------------:|:---------------------------|
| all_signals      | events      |       5219 |        3423 |           34.41 |               25 | False                      |
| all_signals      | bit_toggles |      23998 |       12111 |           49.53 |               25 | False                      |
| internal         | events      |       2399 |         903 |           62.36 |               25 | False                      |
| internal         | bit_toggles |      13114 |        3694 |           71.83 |               25 | False                      |
| internal+outputs | events      |       3001 |        1205 |           59.85 |               25 | False                      |
| internal+outputs | bit_toggles |      16623 |        4736 |           71.51 |               25 | False                      |
| clock            | events      |        823 |         823 |            0    |               25 | False                      |
| clock            | bit_toggles |        823 |         823 |            0    |               25 | False                      |
| inputs           | events      |       1395 |        1395 |            0    |               25 | False                      |
| inputs           | bit_toggles |       6552 |        6552 |            0    |               25 | False                      |

## Sweep: internal-register bit toggles

|   valid_duty_pct | inputs_held_when_invalid   |   baseline |   optimized |   reduction_pct |
|-----------------:|:---------------------------|-----------:|------------:|----------------:|
|            100   | False                      |      12520 |       12520 |            0    |
|             50   | False                      |      13714 |        7442 |           45.73 |
|             25   | False                      |      13114 |        3694 |           71.83 |
|             12.5 | False                      |      12814 |        1860 |           85.48 |
|              6.2 | False                      |      12664 |         941 |           92.57 |
|            100   | True                       |      12520 |       12520 |            0    |
|             50   | True                       |       7470 |        7470 |            0    |
|             25   | True                       |       3769 |        3769 |            0    |
|             12.5 | True                       |       1925 |        1925 |            0    |
|              6.2 | True                       |        924 |         924 |            0    |

## Yosys generic synthesis

| design        |   cells |   flops |   enable_flops |
|:--------------|--------:|--------:|---------------:|
| mac_baseline  |     546 |      84 |              0 |
| mac_optimized |     546 |      84 |             81 |

`enable_flops` > 0 means the valid gating was mapped to enable flip-flops (hold muxes), not clock-gating cells.

Switching activity is a proxy for dynamic power. No capacitance model, no gate-level power analysis.
