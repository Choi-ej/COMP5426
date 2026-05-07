# COMP5426 : Parallel and Distributed Computing
# Assignment 1-Parallel Matrix Multiplication Optimisation with Pthreads

Designed and implemented a high-performance parallel matrix multiplication system in C using POSIX threads (`pthreads`) as part of **COMP4426/5426 Parallel and Distributed Computing** at the University of Sydney. The project focused on analysing execution time, scalability, RAM usage, and low-level optimisation techniques for large-scale matrix operations. :contentReference[oaicite:0]{index=0}

## Key Features
- Parallel matrix multiplication using pthreads
- Dynamic multiplication-order optimisation:
  - `(A × B) × C`
  - `A × (B × C)`
- 1D row-wise workload partitioning
- Sequential vs parallel performance benchmarking
- Loop unrolling optimisation (factor 4)
- Float vs double precision analysis
- Execution time, speedup, and RAM profiling

## Optimisation Techniques
- Multi-threaded computation
- Cache-friendly memory access patterns
- Instruction-level parallelism (ILP)
- Loop unrolling on nested loops
- Pointer arithmetic optimisation
- Cost-based computation ordering

## Tools & Technologies
- C
- POSIX Threads (pthreads)
- GCC / Clang
- macOS ARM64 (Apple M1)
- Shell scripting
- Performance benchmarking & profiling

## Performance Results
- Achieved over **14× speedup** using loop unrolling and multithreading
- Significant reduction in execution time compared to sequential implementation
- Analysed scalability limits caused by CPU core count, cache efficiency, and memory bandwidth

## Skills Demonstrated
Parallel programming, performance optimisation, multithreading, low-level systems programming, cache optimisation, benchmarking, algorithm analysis, and concurrent computing.
