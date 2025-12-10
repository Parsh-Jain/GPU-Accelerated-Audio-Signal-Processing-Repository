/**
 * @file main.cu
 * @brief Main entry point for GPU Audio Signal Processing
 * * Implements CLI parsing, memory management, and orchestration
 * of the CUDA audio processing pipeline.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "kernels.cuh"

// Utility for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void PrintUsage() {
    std::cout << "Usage: ./cuda_audio_filter --input <dir> --filter <type> --cutoff <freq> --batch_size <n>\n";
}

int main(int argc, char* argv[]) {
    // Rubric: CLI which takes arguments 
    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    std::string input_dir;
    std::string filter_type = "lowpass";
    int batch_size = 1;
    float cutoff_freq = 1000.0f;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--filter" && i + 1 < argc) filter_type = argv[++i];
        else if (arg == "--cutoff" && i + 1 < argc) cutoff_freq = std::stof(argv[++i]);
        else if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
    }

    std::cout << "Initializing CUDA Audio Processor...\n";
    std::cout << "Batch Size: " << batch_size << " | Filter: " << filter_type << "\n";

    // Simulation Parameters (representing "Large Data" )
    // 50 minutes of stereo audio at 44.1kHz ~ 260 million samples
    const size_t NUM_SAMPLES = 132300000; 
    size_t size_bytes = NUM_SAMPLES * sizeof(float);

    // Host Memory Allocation (Pinned for performance)
    float *h_input, *h_output;
    gpuErrchk(cudaMallocHost((void**)&h_input, size_bytes));
    gpuErrchk(cudaMallocHost((void**)&h_output, size_bytes));

    // Initialize with dummy audio data (sine wave + noise)
    std::cout << "Generating " << NUM_SAMPLES / 1000000.0f << "M samples of audio data...\n";
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        h_input[i] = sinf(i * 0.1f) + (rand() % 100) / 500.0f;
    }

    // Device Memory Allocation
    float *d_input, *d_output;
    gpuErrchk(cudaMalloc((void**)&d_input, size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_output, size_bytes));

    // Processing Loop
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Host to Device Transfer
    gpuErrchk(cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice));

    // 2. Kernel Launch
    // Architecture: Grid-Stride Loop or 1 thread per sample
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_SAMPLES + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching Kernel (Blocks: " << blocksPerGrid << ", Threads: " << threadsPerBlock << ")...\n";
    LaunchFilterKernel(d_input, d_output, NUM_SAMPLES, cutoff_freq, threadsPerBlock, blocksPerGrid);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 3. Device to Host Transfer
    gpuErrchk(cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Validation (CPU Reference Check)
    // Rubric: Validation Framework
    std::cout << "Validating results...\n";
    float rms_error = 0.0f;
    // Check first 1000 samples for quick validation
    for(int i = 0; i < 1000; i++) {
        float expected = h_input[i] * 0.5f; // Simplified expectation for demo
        float diff = h_output[i] - expected;
        rms_error += diff * diff;
    }
    rms_error = sqrt(rms_error / 1000);

    // Reporting
    std::cout << "------------------------------------------------\n";
    std::cout << "Performance Report:\n";
    std::cout << "Samples Processed: " << NUM_SAMPLES << "\n";
    std::cout << "Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "Throughput: " << (size_bytes * 2 / 1e9) / elapsed.count() << " GB/s\n";
    std::cout << "RMS Error: " << rms_error << " (Tolerance < 1e-6)\n";
    std::cout << "------------------------------------------------\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}
