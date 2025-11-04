# Parallel Image Filtering with CUDA Capstone Project

A comprehensive implementation of parallel computing techniques for high-performance image processing using CUDA architecture

## Project Overview

This project demonstrates the practical application of GPU parallelism for image processing through a complete implementation of image filters in CUDA/PyCUDA. The system provides empirical evidence of performance advantages, achieving 3.5-4.2× acceleration over sequential CPU processing across three fundamental image operations.

## Core Capabilities

- **Parallel Filter Implementation**: CUDA-optimized Gaussian blur, sharpening, and edge detection algorithms
- **Performance Benchmarking**: Direct CPU vs GPU execution time comparison framework
- **Command Line Interface**: Efficient batch processing interface
- **Multiple Filter Operations**: Comprehensive support for blur, sharpen, and edge detection
- **Performance Analytics**: Detailed timing metrics and acceleration ratios
- **Production Code Standards**: Enterprise-grade implementation with comprehensive error handling

## Google Colab Quick Start

### Environment Configuration

1. Create a new Google Colab notebook
2. Access Runtime → Change runtime type
3. Select GPU as Hardware accelerator
4. Apply configuration

### System Verification

```bash
!nvidia-smi
!ls /usr/lib/x86_64-linux-gnu/libcuda*
```

### Project Setup

```bash
!git clone https://github.com/Ritviek/Parallel-Image-Filtering-with-CUDA-Capstone-Project.git
%cd /content/Parallel-Image-Filtering-with-CUDA-Capstone-Project
```

### Package Installation

```bash
!pip install -r requirements.txt
```

### Filter Processing Examples

**Gaussian Blur Application:**
```bash
!python main.py samples/landscape.jpg outputs/landscape_blur.jpg --filter blur
```

**Image Sharpening Processing:**
```bash
!python main.py samples/nature.jpg outputs/nature_sharpen.jpg --filter sharpen
```

**Edge Detection Execution:**
```bash
!python main.py samples/nature.jpg outputs/nature_edge.jpg --filter edge
```

### Performance Evaluation

```bash
!python main.py samples/landscape.jpg outputs/landscape_comparison.jpg --filter blur --compare-cpu --verbose
```

### Results Visualization

```bash
# Display processed outputs
!ls -la outputs/

# Comparative visualization
from IPython.display import Image, display
import matplotlib.pyplot as plt
import cv2

# Load image pairs
source_image = cv2.imread('samples/landscape.jpg')
result_image = cv2.imread('outputs/landscape_blur.jpg')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('GPU Processed (Blur Filter)')
plt.axis('off')
plt.show()
```

## System Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 10.0 or later (pre-configured in Google Colab)
- Minimum 2GB GPU memory recommended

## Required Packages

- python>=3.7
- numpy>=1.19.0
- opencv-python>=4.5.0
- pycuda>=2020.1

## Usage Guide

### Basic Command Structure

```bash
python main.py input_image.jpg output_image.jpg --filter blur
```

### Available Filter Types

- **blur**: 5x5 Gaussian blur convolution
- **sharpen**: Image sharpening enhancement
- **edge**: Sobel-based edge detection

### Command Parameters

- `input`: Source image file path
- `output`: Destination image file path
- `--filter`: Processing algorithm selection (blur, sharpen, edge)
- `--compare-cpu`: Enable performance comparison mode
- `--verbose`: Activate detailed processing logs

## Performance Metrics

Benchmark results on NVIDIA Tesla T4 (Google Colab environment):

| Resolution  | Operation   | CPU Duration | GPU Duration | Acceleration |
|-------------|-------------|--------------|--------------|--------------|
| 1920x1080   | Blur        | 0.045s       | 0.012s       | 3.75×        |
| 1920x1080   | Sharpen     | 0.038s       | 0.009s       | 4.22×        |
| 1920x1080   | Edge        | 0.052s       | 0.014s       | 3.71×        |

*Performance may vary based on GPU specifications and image properties*

## Test Image Collection

The framework includes 5 sample images for evaluation:

- `samples/landscape.jpg` - Scenic landscape photography
- `samples/portrait.jpg` - Portrait composition
- `samples/architecture.jpg` - Structural architecture
- `samples/nature.jpg` - Natural environment scene
- `samples/abstract.jpg` - Abstract visual patterns

## Technical Architecture

### CUDA Kernel Implementations

Three specialized GPU kernels:

1. **Gaussian Blur Kernel**: 5×5 Gaussian convolution processing
2. **Sharpening Kernel**: 3×3 sharpening convolution matrix
3. **Edge Detection Kernel**: Sobel edge detection implementation

### Memory Architecture

- Optimized GPU memory allocation and release
- Efficient host-device data transfer protocols
- Advanced boundary management for convolution operations

### Performance Enhancements

- 16×16 thread block organization for maximum GPU throughput
- Coalesced memory access optimization
- Reduced host-device transfer overhead

## Repository Organization

```
Parallel-Image-Filtering-with-CUDA-Capstone-Project/
├── main.py              # Primary application with CUDA kernels
├── README.md           # Project documentation
├── requirements.txt    # Python package dependencies
├── run.sh             # Automated execution script
├── Makefile           # Build configuration
├── create_samples.py  # Sample image creation utility
├── samples/           # Test image directory
│   ├── landscape.jpg
│   ├── portrait.jpg
│   ├── architecture.jpg
│   ├── nature.jpg
│   └── abstract.jpg
└── outputs/           # Processed image results
```

## Algorithm Specifications

### Gaussian Blur Implementation

5×5 Gaussian kernel with sigma approximation:
- Weight matrix: [1,4,7,4,1; 4,16,26,16,4; 7,26,41,26,7; 4,16,26,16,4; 1,4,7,4,1] / 273

### Sharpening Algorithm

Standard 3×3 sharpening convolution:
- Kernel: [0,-1,0; -1,5,-1; 0,-1,0]

### Edge Detection Methodology

Sobel operator with gradient computation:
- Sobel X: [-1,0,1; -2,0,2; -1,0,1]
- Sobel Y: [-1,-2,-1; 0,0,0; 1,2,1]

## Issue Resolution

### Common Challenges

**CUDA Detection Failure:**
```
Issue: CUDA unavailable
Resolution: Verify GPU runtime selection in Colab
```

**Memory Capacity Exceeded:**
```
Issue: CUDA memory exhaustion
Resolution: Decrease image dimensions or restart environment
```

**PyCUDA Import Failure:**
```
Issue: Missing pycuda module
Resolution: Execute !pip install pycuda
```

## Technical Implementation Details

The project employs data-parallel computing principles to distribute image processing workloads across hundreds of GPU cores simultaneously. Each pixel operation is mapped to individual threads, enabling massive parallelism that significantly outperforms traditional sequential CPU processing approaches.

## License & Acknowledgments

This capstone project demonstrates practical GPU programming concepts for high-performance computing applications in image processing and computer vision.
```
