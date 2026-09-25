# Nsight evidence — measured DRAM traffic

Achievable DRAM bandwidth (2 GiB device-to-device copy): **1765 GB/s**.

| target                              |   kernels |   wall_ms |   kernel_time_ms |   gpu_busy_fraction |   dram_total_mb |   model_estimate_mb |   measured_over_model |   dram_gbps_over_kernel_time |   dram_gbps_over_wall_time |   pct_of_achievable_over_wall |   dram_pct_peak_time_weighted |   sm_pct_peak_time_weighted |
|:------------------------------------|----------:|----------:|-----------------:|--------------------:|----------------:|--------------------:|----------------------:|-----------------------------:|---------------------------:|------------------------------:|------------------------------:|----------------------------:|
| decode_tinyllama_b32_ctx1024_static |      1192 |    26.925 |           24.811 |               0.921 |        16313.1  |             2938.29 |                 5.552 |                      657.496 |                    605.875 |                        34.323 |                        32.265 |                      44.39  |
| decode_tinyllama_b1_ctx1024_static  |      1082 |    26.022 |            7.714 |               0.296 |         2340.68 |             2223.16 |                 1.053 |                      303.422 |                     89.95  |                         5.096 |                        14.914 |                      10.291 |

`kernel_time_ms` is the sum of ncu-measured kernel durations (kernels serialized by ncu); `wall_ms` is the same iteration timed without the profiler. `gpu_busy_fraction` = kernel time / wall time. `dram_*_pct` are ncu's own percent-of-peak counters, weighted by kernel time.

Top kernels by DRAM bytes:

- **decode_tinyllama_b32_ctx1024_static**: fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMem (296 MB); fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMem (296 MB); fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMem (296 MB); fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMem (296 MB); fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMem (296 MB)
- **decode_tinyllama_b1_ctx1024_static**: ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages (134 MB); std::enable_if<!T7, void>::type internal::kernel<i (24 MB); std::enable_if<!T7, void>::type internal::kernel<i (24 MB); std::enable_if<!T7, void>::type internal::kernel<i (24 MB); std::enable_if<!T7, void>::type internal::kernel<i (24 MB)
