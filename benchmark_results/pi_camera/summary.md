# SmartLock Benchmark Summary

- Source: `camera /dev/video0`
- Frames measured: 489
- Measurement wall time: 60.05 s
- Throughput: 8.14 FPS

| Metric | No face | Face |
|---|---:|---:|
| Face detection latency | 111.26 ms | 108.43 ms |
| Face recognition latency | 0.00 ms | 2.15 ms |
| Fuzzy decision latency | 0.08 ms | 5.90 ms |
| JPEG/base64 encode latency | 5.91 ms | 6.34 ms |
| Total pipeline latency | 121.36 ms | 127.36 ms |
| RAM usage | 89.27 MB | 89.25 MB |
| CPU utilization | 319.07% | 319.07% |
