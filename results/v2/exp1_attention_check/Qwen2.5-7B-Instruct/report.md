# Prefill attention backend and memory — Qwen/Qwen2.5-7B-Instruct

## Part 1 — attention-related kernels in one prefill (4 × 2048, Exp 1 code path)

| kernel                                                                                                         |   calls |   total_us |
|:---------------------------------------------------------------------------------------------------------------|--------:|-----------:|
| void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::half_t, F |      28 |    20533.2 |

**Backend: FlashAttention.** Layers: 28.

## Part 2 — prefill peak memory (above weights) and TTFT; chunk = 1024 tokens

|   batch |   prompt |   tokens_in_flight | mode    | status   |   ttft_ms |   peak_extra_gib |
|--------:|---------:|-------------------:|:--------|:---------|----------:|-----------------:|
|       4 |     2048 |               8192 | full    | ok       |    597.97 |             2.81 |
|       4 |     2048 |               8192 | last    | ok       |    568.97 |             1.53 |
|       4 |     2048 |               4096 | chunked | ok       |    674.8  |             0.99 |
|      16 |     4096 |              65536 | full    | ok       |   4902.16 |            22.5  |
|      16 |     4096 |              65536 | last    | ok       |   4633.41 |            12.22 |
|      16 |     4096 |              16384 | chunked | ok       |   5240.98 |             5.75 |
|      32 |     2048 |              65536 | full    | ok       |   4813.3  |            22.5  |
|      32 |     2048 |              65536 | last    | ok       |   4527.43 |            12.22 |
|      32 |     2048 |              32768 | chunked | ok       |   4780.07 |             7.93 |
|      32 |     4096 |             131072 | full    | ok       |  10055.1  |            45    |
|      32 |     4096 |             131072 | last    | ok       |   9959.16 |            24.44 |
|      32 |     4096 |              32768 | chunked | ok       |  10540.4  |            11.5  |
|      16 |     8192 |             131072 | full    | ok       |  10614.4  |            45    |
|      16 |     8192 |             131072 | last    | ok       |  10487.2  |            24.44 |
|      16 |     8192 |              16384 | chunked | ok       |  12249.2  |             9.69 |
|      32 |     8192 |             262144 | full    | OOM      |    nan    |           nan    |
|      32 |     8192 |             262144 | last    | ok       |  19614.3  |            48.88 |
|      32 |     8192 |              32768 | chunked | ok       |  24484.5  |            19.38 |

`tokens_in_flight` = tokens processed by one forward pass (batch × prompt, or batch × chunk).
