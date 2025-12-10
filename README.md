# GPU-Accelerated-Audio-Signal-Processing-Repository

# GPU-Accelerated Audio Signal Processing at Scale

## Project Overview
This project implements GPU-accelerated audio signal processing using CUDA, demonstrating efficient batch processing of large audio datasets on modern NVIDIA GPUs. [cite_start]The implementation leverages CUDA kernels to perform real-time digital signal processing filters including lowpass, highpass, and bandpass filters[cite: 18].

### Key Components
* **CUDA Kernels:** Custom kernels implementing Butterworth filter algorithms.
* **Memory Management:** Optimized pinned host memory for faster H2D/D2H transfers.
* **Data Pipeline:** Efficient batching system for processing multiple gigabytes of audio.
* **Validation:** CPU reference implementation for correctness verification.

## Build and Run Instructions

### Prerequisites
* NVIDIA GPU (SM 7.0+)
* CUDA Toolkit (11.0+)
* C++ Compiler (GCC/G++)
* Make

### Compilation
To build the project, run:
```bash
make
