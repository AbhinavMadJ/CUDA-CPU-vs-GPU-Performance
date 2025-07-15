# CUDA Programming: CPU vs GPU Performance

This mini-project demonstrates the performance difference between CPU and GPU execution by computing a nonlinear mathematical function over **10 million floating-point values**.

The function evaluated is:

\[
y = sin(x) + log(x + 1) + sqrt{x}
\]


## Features
- CPU implementation in C++
- GPU implementation in CUDA
- Performance benchmarking and timing comparison
- Run and tested on Google Colab with NVIDIA Tesla T4 GPU


## Repository Contents

| File | Description |
|------|-------------|
| `cpu_nonlinear.cpp` | C++ source code for CPU implementation |
| `gpu_nonlinear.cu` | CUDA source code for GPU implementation |
| `HPC_CUDA.ipynb` | Colab notebook containing both implementations and results |
| `AHP_CUDA_Report.pdf` | Detailed report including setup, results, and inference |


## Tech Stack
- C++
- CUDA Toolkit 11.8
- NVIDIA Tesla T4 GPU (Google Colab)


## How to Run

You need to have **CUDA Toolkit** installed (or use Google Colab with GPU enabled).

### Compile and Run CPU version
Open a terminal in the project directory and run:
```
g++ cpu_nonlinear.cpp -o cpu_exec
./cpu_exec
```

### Compile and Run GPU version
Open a terminal in the project directory and run:
```
nvcc gpu_nonlinear.cu -o gpu_exec
./gpu_exec
```


## Results

| Metric | CPU | GPU |
|--------|-----|-----|
| Time (seconds) | 0.3584 | 0.0506 |

**Configuration:**
- Threads per block: 256  
- Blocks per grid: 39,063  
- Total GPU threads launched: 10,000,128  

**Conclusion:**  
GPU execution achieved approximately **7× speedup** over CPU.
