# SmartLock Benchmark Summary

- Source: `camera /dev/video0`
- Frames measured: 191
- Measurement wall time: 30.11 s
- Throughput: 6.34 FPS

| Metric | No face | Face |
|---|---:|---:|
| Face detection latency | 146.79 ms | 145.06 ms |
| Face recognition latency | 0.00 ms | 2.31 ms |
| Fuzzy decision latency | 0.08 ms | 6.32 ms |
| JPEG/base64 encode latency | 6.18 ms | 6.62 ms |
| Total pipeline latency | 154.94 ms | 163.05 ms |
| RAM usage | 89.11 MB | 89.10 MB |
| CPU utilization | 302.29% | 302.29% |
