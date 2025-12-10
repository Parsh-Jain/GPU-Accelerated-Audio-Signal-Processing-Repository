#!/bin/bash

echo "Building Project..."
make

if [ $? -eq 0 ]; then
    echo "Build Successful. Running Batch Processing Demo..."
    
    # Create a dummy data directory for demonstration if it doesn't exist
    mkdir -p data
    
    # Run the CUDA application
    # Simulating a run with arguments as per Rubric Requirement
    ./bin/cuda_audio_filter --input ./data --filter lowpass --cutoff 440 --batch_size 16
else
    echo "Build Failed."
fi
