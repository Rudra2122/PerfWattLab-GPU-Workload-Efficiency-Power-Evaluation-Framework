# Experiment 2 — KV cache — TinyLlama/TinyLlama-1.1B-Chat-v1.0

## A. Analytical footprint

KV bytes/token = 2 × 22 layers × 4 KV heads × 64 head_dim × 2 B = **22,528 B** (MHA-equivalent with 32 heads: 180,224 B, 8×)

|   context |   gqa_kv_heads |   mha_heads |   gqa_bytes_per_token |   mha_bytes_per_token |   gqa_mib_per_seq |   mha_mib_per_seq |   gqa_seqs_per_gib |   mha_seqs_per_gib |
|----------:|---------------:|------------:|----------------------:|----------------------:|------------------:|------------------:|-------------------:|-------------------:|
|       512 |              4 |          32 |                 22528 |                180224 |                11 |                88 |              93.09 |              11.64 |
|      2048 |              4 |          32 |                 22528 |                180224 |                44 |               352 |              23.27 |               2.91 |

## B. Measured past_key_values bytes vs formula

|   context |   formula_bytes |   measured_past_bytes | match   |   allocator_delta_bytes |   allocator_overhead_ratio |
|----------:|----------------:|----------------------:|:--------|------------------------:|---------------------------:|
|       128 |         2883584 |               2883584 | True    |                11403264 |                    3.95455 |
|       256 |         5767168 |               5767168 | True    |                 5767168 |                    1       |
|       512 |        11534336 |              11534336 | True    |                11534336 |                    1       |
|      1024 |        23068672 |              23068672 | True    |                23068672 |                    1       |
|      1536 |        34603008 |              34603008 | True    |                34603008 |                    1       |
|      2000 |        45056000 |              45056000 | True    |                45056000 |                    1       |

## C. Decode with HF DynamicCache

Step time grows ≈ -22.38 µs per extra context token (linear fit, batch 1). Logical KV grows exactly by bytes/token per step; see growth.csv for the allocator view.

## D. Decode step vs context — HF contiguous vs paged + gather (batch 1)

|   context |   hf_step_ms |   paged_gather_ms |   paged_step_ms_incl_gather |   gather_share |
|----------:|-------------:|------------------:|----------------------------:|---------------:|
|       128 |       22.379 |             0.498 |                      23.824 |          0.021 |
|       256 |       23.506 |             0.518 |                      23.781 |          0.022 |
|       512 |       23.189 |             0.567 |                      23.57  |          0.024 |
|      1024 |       23.003 |             0.771 |                      23.698 |          0.033 |
|      1536 |       22.97  |             0.94  |                      24.179 |          0.039 |
|      2000 |       22.728 |             1.046 |                      23.914 |          0.044 |

## E. Decode step vs batch (context 1024)

|   batch |   context | status   |   step_ms |    tok_s |   kv_bytes_read_mb |   weight_bytes_mb |   kv_share |   est_total_gbps |   pct_peak_bw |
|--------:|----------:|:---------|----------:|---------:|-------------------:|------------------:|-----------:|-----------------:|--------------:|
|       1 |      1024 | ok       |    22.652 |   44.146 |             23.069 |            2200.1 |      0.01  |           98.145 |         6.312 |
|       2 |      1024 | ok       |    24.528 |   81.541 |             46.137 |            2200.1 |      0.021 |           91.58  |         5.889 |
|       4 |      1024 | ok       |    24.665 |  162.171 |             92.275 |            2200.1 |      0.04  |           92.939 |         5.977 |
|       8 |      1024 | ok       |    24.463 |  327.026 |            184.549 |            2200.1 |      0.077 |           97.48  |         6.269 |
|      16 |      1024 | ok       |    25.012 |  639.692 |            369.099 |            2200.1 |      0.144 |          102.718 |         6.606 |
|      32 |      1024 | ok       |    26.921 | 1188.66  |            738.198 |            2200.1 |      0.251 |          109.145 |         7.019 |
|      64 |      1024 | ok       |    29.022 | 2205.21  |           1476.39  |            2200.1 |      0.402 |          126.679 |         8.147 |

## F. Max concurrent sequences (HF contiguous cache, one decode step)

|   context |   free_mem_gib_before |   analytic_kv_only_ceiling |   measured_max_batch | limited_by   |   measured_vs_analytic |
|----------:|----------------------:|---------------------------:|---------------------:|:-------------|-----------------------:|
|       512 |                36.824 |                       3427 |                 1676 | oom          |                  0.489 |
|      1024 |                36.824 |                       1713 |                  844 | oom          |                  0.493 |
|      2000 |                36.824 |                        877 |                  432 | oom          |                  0.493 |

`limited_by`: `oom` = ran out of memory; `launch_limit` = a kernel's launch configuration exceeded CUDA limits first; `cap` = hit --max-batch-cap without failing. Below the KV-only ceiling, the gap is activation/workspace memory plus the copy DynamicCache makes when it appends the new token (old and new cache coexist).

## G. Paged vs contiguous max-length preallocation (heavy-tailed lengths)

|                           |     value |
|:--------------------------|----------:|
| budget_gib                |    29.459 |
| kv_bytes_per_token        | 22528     |
| max_len                   |  2048     |
| block_size                |    16     |
| contiguous_max_concurrent |   685     |
| contiguous_utilization    |     0.186 |
| paged_max_concurrent      |  3675     |
| paged_utilization         |     0.98  |
| paged_internal_frag_slots | 27474     |
