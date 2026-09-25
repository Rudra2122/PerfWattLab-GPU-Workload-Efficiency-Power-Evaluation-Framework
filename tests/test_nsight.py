from run_nsight import METRICS, parse_ncu_csv, summarize_kernels

SAMPLE = '''==PROF== Connected to process 1234 (/usr/bin/python3)
==PROF== Profiling "ampere_fp16_gemm": 0%....50%....100% - 2 passes
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","dram__bytes_read.sum","dram__bytes_write.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed"
"","","","","","","","","","","","byte","byte","%","nsecond","%"
"0","1234","python3","host","ampere_fp16_gemm","1","7","(128, 1, 1)","(80, 1, 1)","0","8.0","2,000,000,000","1,000,000","80.5","2,000,000","30.0"
"1","1234","python3","host","rms_norm_kernel","1","7","(256, 1, 1)","(1, 1, 1)","0","8.0","4096","8192","0.5","3,000","1.0"
==PROF== Disconnected from process 1234
'''


def test_parse_and_summarize(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(SAMPLE)
    df = parse_ncu_csv(p)
    assert len(df) == 2 and list(df.kernel) == ["ampere_fp16_gemm", "rms_norm_kernel"]
    assert df["dram__bytes_read.sum"].iloc[0] == 2e9
    s = summarize_kernels(df)
    assert s["kernels"] == 2
    assert abs(s["dram_total_mb"] - (2e9 + 1e6 + 4096 + 8192) / 1e6) < 1e-6
    assert abs(s["kernel_time_ms"] - 2.003) < 1e-9
    # time-weighted % of peak ≈ dominated by the long kernel
    assert 80 < s["dram_pct_peak_time_weighted"] < 80.5
    assert set(METRICS) <= set(df.columns)
