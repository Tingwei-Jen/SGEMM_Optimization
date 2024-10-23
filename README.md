# SGEMM_Optimization
Running on NVIDIA GeForce RTX 3070 Ti
![](plot.png)
### GFLOPS at 4096x4096:
<!-- benchmark_results -->
| Kernels                              |  GFLOPS  | Performance relative to cuBLAS |
|:-------------------------------------|---------:|:-------------------------------|
| Naive                                |   `1395` | 9.5%                           |
| SMEM Method                          |   `1316` | 9.0%                           |
| 2D Tiling                            |  `6839`  | 46.9%                          |
| Solve Bank Conflicts (Padding)       |  `7776`  | 53.3%                          |
| Register                             | `7522`   | 51.5%                          |
| Float4                               | `12601`  | 86.3%                          |
| Float4 + Prefetch                    | `9006`   | 61.7%                          |
| Tuning                               | `12942`  | 88.7%                          |
| cuBLAS                               | `14597`  | 100.0%                         |
<!-- benchmark_results -->
