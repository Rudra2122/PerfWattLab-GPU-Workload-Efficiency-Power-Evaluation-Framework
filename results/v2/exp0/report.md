# Experiment 0 — pipeline vs direct generate (v2 methodology)

Fixed output length: 128 tokens. Energy method: `counter`. Runs per variant: 60 (interleaved).

## pipeline → direct

- **generation latency (ms)**: pooled -9.7%  (ratio 0.903, 95% CI 0.899–0.905) → significant decrease; paired per-prompt -10.0%  (ratio 0.900, 95% CI 0.898–0.902) → significant decrease
- **ms / output token**: pooled -9.3%  (ratio 0.907, 95% CI 0.904–0.912) → significant decrease; paired per-prompt -9.4%  (ratio 0.906, 95% CI 0.904–0.908) → significant decrease
- **energy / query (J)**: pooled -7.5%  (ratio 0.925, 95% CI 0.924–0.926) → significant decrease; paired per-prompt -8.2%  (ratio 0.918, 95% CI 0.911–0.924) → significant decrease
- **energy / token (J)**: pooled -6.8%  (ratio 0.932, 95% CI 0.931–0.933) → significant decrease; paired per-prompt -7.6%  (ratio 0.924, 95% CI 0.917–0.930) → significant decrease
- **active (above-idle) energy / token (J)**: pooled +1.6%  (ratio 1.016, 95% CI 0.980–1.040) → no significant change (CI contains 1.0); paired per-prompt -0.8%  (ratio 0.992, 95% CI 0.960–1.021) → no significant change (CI contains 1.0)
- **average board power (W)**: pooled +2.0%  (ratio 1.020, 95% CI 1.014–1.025) → significant increase; paired per-prompt +1.6%  (ratio 1.016, 95% CI 1.010–1.022) → significant increase
- **NVML GPU utilization (%)**: pooled +9.7%  (ratio 1.097, 95% CI 1.093–1.100) → significant increase; paired per-prompt +9.7%  (ratio 1.097, 95% CI 1.094–1.101) → significant increase

## direct_no_grad → direct

- **generation latency (ms)**: pooled -9.9%  (ratio 0.901, 95% CI 0.898–0.903) → significant decrease; paired per-prompt -10.1%  (ratio 0.899, 95% CI 0.896–0.902) → significant decrease
- **ms / output token**: pooled -9.9%  (ratio 0.901, 95% CI 0.898–0.903) → significant decrease; paired per-prompt -10.1%  (ratio 0.899, 95% CI 0.896–0.902) → significant decrease
- **energy / query (J)**: pooled -7.5%  (ratio 0.925, 95% CI 0.924–0.926) → significant decrease; paired per-prompt -8.3%  (ratio 0.917, 95% CI 0.911–0.922) → significant decrease
- **energy / token (J)**: pooled -7.5%  (ratio 0.925, 95% CI 0.924–0.926) → significant decrease; paired per-prompt -8.3%  (ratio 0.917, 95% CI 0.911–0.922) → significant decrease
- **active (above-idle) energy / token (J)**: pooled +1.5%  (ratio 1.015, 95% CI 0.977–1.026) → no significant change (CI contains 1.0); paired per-prompt -1.7%  (ratio 0.983, 95% CI 0.955–1.009) → no significant change (CI contains 1.0)
- **average board power (W)**: pooled +2.2%  (ratio 1.022, 95% CI 1.015–1.024) → significant increase; paired per-prompt +1.6%  (ratio 1.016, 95% CI 1.010–1.022) → significant increase
- **NVML GPU utilization (%)**: pooled +9.5%  (ratio 1.095, 95% CI 1.091–1.100) → significant increase; paired per-prompt +9.6%  (ratio 1.096, 95% CI 1.092–1.099) → significant increase

## pipeline → direct_no_grad

- **generation latency (ms)**: pooled +0.2%  (ratio 1.002, 95% CI 0.999–1.004) → no significant change (CI contains 1.0); paired per-prompt +0.1%  (ratio 1.001, 95% CI 0.998–1.005) → no significant change (CI contains 1.0)
- **ms / output token**: pooled +0.7%  (ratio 1.007, 95% CI 1.005–1.011) → significant increase; paired per-prompt +0.8%  (ratio 1.008, 95% CI 1.005–1.011) → significant increase
- **energy / query (J)**: pooled -0.0%  (ratio 1.000, 95% CI 0.999–1.001) → no significant change (CI contains 1.0); paired per-prompt +0.1%  (ratio 1.001, 95% CI 0.995–1.008) → no significant change (CI contains 1.0)
- **energy / token (J)**: pooled +0.8%  (ratio 1.008, 95% CI 1.006–1.009) → significant increase; paired per-prompt +0.8%  (ratio 1.008, 95% CI 1.002–1.014) → significant increase
- **active (above-idle) energy / token (J)**: pooled +0.1%  (ratio 1.001, 95% CI 0.989–1.026) → no significant change (CI contains 1.0); paired per-prompt +1.0%  (ratio 1.010, 95% CI 0.980–1.038) → no significant change (CI contains 1.0)
- **average board power (W)**: pooled -0.1%  (ratio 0.999, 95% CI 0.997–1.003) → no significant change (CI contains 1.0); paired per-prompt +0.0%  (ratio 1.000, 95% CI 0.995–1.005) → no significant change (CI contains 1.0)
- **NVML GPU utilization (%)**: pooled +0.2%  (ratio 1.002, 95% CI 0.996–1.006) → no significant change (CI contains 1.0); paired per-prompt +0.1%  (ratio 1.001, 95% CI 0.997–1.006) → no significant change (CI contains 1.0)

## Aggregate energy (all interleaved queries)

| variant        |   energy_j |   tokens |   seconds |   j_per_token |   avg_power_w |
|:---------------|-----------:|---------:|----------:|--------------:|--------------:|
| direct         |    12220.1 |     7680 |   179.95  |        1.5912 |       67.9084 |
| direct_no_grad |    13353.4 |     7680 |   199.804 |        1.7387 |       66.8325 |
| pipeline       |    13305.6 |     7731 |   199.326 |        1.7211 |       66.7527 |

## Roofline check (batch-1 decode)

Weights ≈ 2.20 GB; peak BW ≈ 1555 GB/s → bandwidth floor ≈ **1.4 ms/token**.

- pipeline: median 25.8 ms/token = 18.2× the floor
- direct: median 23.4 ms/token = 16.5× the floor
- direct_no_grad: median 26.0 ms/token = 18.3× the floor

ms/token here includes prefill amortized over output tokens; Experiment 1 measures decode ITL directly.

## H1 — inference_mode vs no_grad (same direct path)

direct(inference_mode) vs direct(no_grad), ms/token: -10.1%  (ratio 0.899, 95% CI 0.896–0.902) → significant decrease

## H2 — pipeline stage breakdown (median ms)

|                |       0 |
|:---------------|--------:|
| preprocess_ms  |    0.92 |
| forward_ms     | 3313.66 |
| postprocess_ms |    0.26 |
| total_ms       | 3314.82 |

Pipeline inference context: `no_grad`

## H3 — generate() kwargs that differ between paths

- `generation_config`: {"pipeline": {"max_length": 2048, "min_length": 0, "early_stopping": false, "do_sample": true, "num_beams": 1, "use_cache": true, "temperature": 0.7, "top_k": 50, "top_p": 1.0, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "remove_invalid_values": false, "num_return_sequences": 1, "output_scores": false, "return_dict_in_generate": false, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "encoder_no_repeat_ngram_size": 0, "num_assistant_tokens": 20, "num_assistant_tokens_schedule": "constant", "assistant_confidence_threshold": 0.4, "assistant_lookbehind": 10, "target_lookbehind": 10, "diversity_penalty": 0.0, "num_beam_groups": 1, "transformers_version": "5.16.1"}, "direct": "<absent>", "direct_no_grad": "<absent>"}
- `use_cache`: {"pipeline": "<absent>", "direct": "True", "direct_no_grad": "True"}

## H4 — run order

- interleaved pipeline→direct ms/token: -9.3%  (ratio 0.907, 95% CI 0.904–0.912) → significant decrease
- blocked (v1-style) pipeline→direct ms/token: -9.2%  (ratio 0.908, 95% CI 0.905–0.913) → significant decrease

Median ms/token by position within each interleaved round:

| variant        |     0 |     1 |     2 |
|:---------------|------:|------:|------:|
| direct         | 23.41 | 23.36 | 23.41 |
| direct_no_grad | 25.98 | 26.01 | 25.92 |
| pipeline       | 25.73 | 25.66 | 25.87 |

## Profiler evidence (one prompt, profiled run)

|                     |   pipeline |    direct |   direct_no_grad |
|:--------------------|-----------:|----------:|-----------------:|
| wall_ms_profiled    |  5418.76   | 4897.49   |        5310.58   |
| gpu_busy_fraction   |     0.1399 |    0.1547 |           0.1426 |
| kernels_per_token   |   948.36   |  955.77   |         955.72   |
| gap_p50_us          |    26.02   |   24.13   |          25.79   |
| gap_total_ms        |  4654.97   | 4136.16   |        4548.14   |
| cpu_ops_per_token   |  3898.1    | 3928.1    |        3928.5    |
| launches_per_token  |   947.51   |  954.91   |         954.91   |
| aten_copy_per_token |   104.61   |  105.42   |         105.42   |

Sync API counts: {"pipeline": {"cudaDeviceSynchronize": 3, "cudaStreamSynchronize": 265, "cudaEventSynchronize": 0}, "direct": {"cudaDeviceSynchronize": 3, "cudaStreamSynchronize": 264, "cudaEventSynchronize": 0}, "direct_no_grad": {"cudaDeviceSynchronize": 3, "cudaStreamSynchronize": 264, "cudaEventSynchronize": 0}}
