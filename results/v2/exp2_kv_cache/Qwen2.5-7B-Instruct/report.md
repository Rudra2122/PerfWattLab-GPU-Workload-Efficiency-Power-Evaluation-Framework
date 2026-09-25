# Experiment 2 — KV cache — Qwen/Qwen2.5-7B-Instruct

## A. Analytical footprint

KV bytes/token = 2 × 28 layers × 4 KV heads × 128 head_dim × 2 B = **57,344 B** (MHA-equivalent with 28 heads: 401,408 B, 7×)

|   context |   gqa_kv_heads |   mha_heads |   gqa_bytes_per_token |   mha_bytes_per_token |   gqa_mib_per_seq |   mha_mib_per_seq |   gqa_seqs_per_gib |   mha_seqs_per_gib |
|----------:|---------------:|------------:|----------------------:|----------------------:|------------------:|------------------:|-------------------:|-------------------:|
|       512 |              4 |          28 |                 57344 |                401408 |                28 |               196 |              36.57 |               5.22 |
|      2048 |              4 |          28 |                 57344 |                401408 |               112 |               784 |               9.14 |               1.31 |
|     32768 |              4 |          28 |                 57344 |                401408 |              1792 |             12544 |               0.57 |               0.08 |

## B. Measured past_key_values bytes vs formula

|   context |   formula_bytes |   measured_past_bytes | match   |   allocator_delta_bytes |   allocator_overhead_ratio |
|----------:|----------------:|----------------------:|:--------|------------------------:|---------------------------:|
|       512 |        29360128 |              29360128 | True    |                38928384 |                    1.32589 |
|      2048 |       117440512 |             117440512 | True    |               124125184 |                    1.05692 |
|      4096 |       234881024 |             234881024 | True    |               235798528 |                    1.00391 |
|      8192 |       469762048 |             469762048 | True    |               469762048 |                    1       |

## C. Decode with HF DynamicCache

Step time grows ≈ -30.68 µs per extra context token (linear fit, batch 1). Logical KV grows exactly by bytes/token per step; see growth.csv for the allocator view.

## D. Decode step vs context — HF contiguous vs paged + gather (batch 1)

|   context |   hf_step_ms |   paged_gather_ms |   paged_step_ms_incl_gather |   gather_share |
|----------:|-------------:|------------------:|----------------------------:|---------------:|
|       512 |       31.078 |             0.675 |                      31.73  |          0.021 |
|      2048 |       31.346 |             1.443 |                      32.953 |          0.044 |
|      4096 |       31.428 |             2.688 |                      34.131 |          0.079 |
|      8192 |       32.081 |             5.251 |                      37.39  |          0.14  |

## E. Decode step vs batch (context 2048)

|   batch |   context | status   |   step_ms |   tok_s |   kv_bytes_read_mb |   weight_bytes_mb |   kv_share |   est_total_gbps |   pct_peak_bw |
|--------:|----------:|:---------|----------:|--------:|-------------------:|------------------:|-----------:|-----------------:|--------------:|
|       1 |      2048 | ok       |    31.422 |  31.825 |            117.441 |           15231.2 |      0.008 |          488.47  |        31.413 |
|       2 |      2048 | ok       |    31.411 |  63.671 |            234.881 |           15231.2 |      0.015 |          492.373 |        31.664 |
|       4 |      2048 | ok       |    32.979 | 121.29  |            469.762 |           15231.2 |      0.03  |          476.094 |        30.617 |
|       8 |      2048 | ok       |    35.305 | 226.594 |            939.524 |           15231.2 |      0.058 |          458.024 |        29.455 |
|      16 |      2048 | ok       |    38.389 | 416.79  |           1879.05  |           15231.2 |      0.11  |          445.712 |        28.663 |
|      32 |      2048 | ok       |    45.999 | 695.663 |           3758.1   |           15231.2 |      0.198 |          412.818 |        26.548 |
|      64 |      2048 | ok       |    67.931 | 942.127 |           7516.19  |           15231.2 |      0.33  |          334.859 |        21.534 |

## F. Max concurrent sequences (HF contiguous cache, one decode step)

|   context |   free_mem_gib_before |   analytic_kv_only_ceiling |   measured_max_batch | limited_by   |   measured_vs_analytic |
|----------:|----------------------:|---------------------------:|---------------------:|:-------------|-----------------------:|
|      2048 |                24.626 |                        225 |                  111 | oom          |                  0.493 |
|      8192 |                24.626 |                         56 |                   27 | oom          |                  0.482 |

`limited_by`: `oom` = ran out of memory; `launch_limit` = a kernel's launch configuration exceeded CUDA limits first; `cap` = hit --max-batch-cap without failing. Below the KV-only ceiling, the gap is activation/workspace memory plus the copy DynamicCache makes when it appends the new token (old and new cache coexist).

## G. Paged vs contiguous max-length preallocation (heavy-tailed lengths)

|                           |     value |
|:--------------------------|----------:|
| budget_gib                |    19.701 |
| kv_bytes_per_token        | 57344     |
| max_len                   | 32768     |
| block_size                |    16     |
| contiguous_max_concurrent |    11     |
| contiguous_utilization    |     0.012 |
| paged_max_concurrent      |   969     |
| paged_utilization         |     0.98  |
| paged_internal_frag_slots |  7189     |
