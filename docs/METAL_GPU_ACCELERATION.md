# Metal GPU Acceleration for ManifoldEM

ManifoldEM now supports GPU acceleration on Apple Silicon Macs using Apple's [MLX framework](https://github.com/ml-explore/mlx). This can provide significant speedups for computationally intensive operations, particularly during distance calculations.

## Overview

Metal GPU acceleration is automatically available on Apple Silicon Macs (M1, M2, M3, M4, etc.) when the MLX library is installed. The implementation provides:

- **Automatic fallback to CPU**: If Metal is not available or disabled, ManifoldEM will seamlessly fall back to CPU computation
- **No code changes required**: Existing workflows work identically with GPU acceleration
- **Configurable**: Can be enabled/disabled via the `use_metal_gpu` parameter

## What Gets Accelerated

The Metal backend accelerates the following operations in the distance calculation step:

1. **2D FFT operations** (`fft2` and `ifft2`)
   - Image filtering in Fourier space
   - Wiener filter applications

2. **Distance matrix computations**
   - Large matrix multiplications
   - Complex conjugate operations
   - The main computational bottleneck in `calc_distance`

Performance improvements depend on:
- Number of particles per projection direction bin (best with >50 particles)
- Image size (number of pixels, best with >64x64)
- GPU model (M1 vs M2 vs M3/M4)
- Data transfer overhead (GPU memory ↔ CPU memory)

**Note**: For small particle counts (<10 particles) or small images, CPU may be faster due to GPU transfer overhead.

## Installation

### Installing MLX

To enable Metal GPU acceleration, install the `metal` optional dependency:

```bash
# Install ManifoldEM with Metal GPU support
pip install -e ".[metal]"
```

Or install MLX separately:

```bash
pip install mlx>=0.20.0
```

### System Requirements

- **macOS** with Apple Silicon (M1, M2, M3, M4, or later)
- **Python 3.9+**
- **MLX 0.20.0 or later**

Metal GPU acceleration will **not** be available on:
- Intel-based Macs
- Linux systems
- Windows systems

On these systems, ManifoldEM will automatically fall back to CPU computation.

## Usage

### Enabling Metal GPU Acceleration

Metal GPU acceleration is **enabled by default** when available. ManifoldEM will automatically detect if you're running on Apple Silicon with MLX installed.

You can verify Metal is available by checking the log output during `calc-distance`:

```
Computing the distances...
Using Metal GPU acceleration (Apple Silicon)
```

### Disabling Metal GPU Acceleration

To disable Metal GPU acceleration and force CPU computation:

1. **Via Python API**:
```python
from ManifoldEM.params import params

params.use_metal_gpu = False
params.save()
```

2. **Via TOML configuration file**:
```toml
# In your project's TOML file
use_metal_gpu = false
```

### Validating Metal GPU Results

To verify that Metal GPU produces the same results as CPU computation, you can enable validation mode:

```python
from ManifoldEM.metal_backend import get_backend

backend = get_backend()
# This will compare GPU results with CPU and warn if they differ significantly
result = backend.compute_distance_matrices(fourier_images, CTF, validate=True)
```

This is useful for verifying numerical accuracy after updates or debugging issues.

### Checking Metal Availability

```python
from ManifoldEM.metal_backend import is_metal_available

if is_metal_available():
    print("Metal GPU acceleration is available!")
else:
    print("Metal GPU not available, using CPU fallback")
```

## Performance Considerations

### When Metal GPU Helps Most

Metal acceleration provides the most benefit when:

- **Large particle counts per bin** (>50 particles)
- **Large image sizes** (>64x64 pixels)
- **Multiple projection direction bins** to process

For very small particle counts (<10 particles), CPU may actually be faster due to GPU transfer overhead.

### Memory Usage

The Metal backend transfers data between CPU and GPU memory. While Apple Silicon's unified memory architecture minimizes this overhead, very large datasets may still see some memory pressure.

If you experience memory issues:
1. Reduce `ncpu` to process fewer bins in parallel
2. Disable Metal GPU acceleration: `params.use_metal_gpu = False`

### Benchmarking

To compare Metal GPU vs CPU performance on your data:

```python
import time
from ManifoldEM.params import params
from ManifoldEM import calc_distance

# Test with Metal GPU
params.use_metal_gpu = True
start = time.time()
calc_distance.op()
gpu_time = time.time() - start
print(f"Metal GPU time: {gpu_time:.2f}s")

# Test with CPU
params.use_metal_gpu = False
start = time.time()
calc_distance.op()
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.2f}s")

print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

## Technical Details

### Implementation

ManifoldEM uses Apple's [MLX framework](https://ml-explore.github.io/mlx/build/html/index.html), which provides:

- NumPy-compatible API for array operations
- Automatic GPU execution on Apple Silicon
- Unified memory architecture (no explicit CPU↔GPU copies)
- Lazy evaluation for optimized computation graphs

### Accelerated Operations

The `ManifoldEM.metal_backend` module provides GPU-accelerated implementations of:

- `fft2()` and `ifft2()` - 2D Fast Fourier Transforms
- `batch_fft2_filter()` - Batch FFT filtering for multiple images (reduces transfer overhead)
- `matmul()` - Matrix multiplication
- `compute_distance_matrices()` - Specialized distance matrix computation with optional validation
- `abs_squared()`, `conj()`, `real()` - Complex array operations

All operations automatically fall back to NumPy/SciPy if Metal is unavailable.

### Thread Safety

The Metal backend is designed to be thread-safe:
- No global state is mutated during computation
- The `use_metal_gpu` parameter is passed explicitly to worker functions
- Safe for use with Python's `multiprocessing.Pool`
- Each worker process has independent backend state

### Numerical Validation

The `compute_distance_matrices()` function supports an optional `validate=True` parameter that:
- Computes results on both GPU and CPU
- Compares results and warns if relative difference exceeds 1e-6
- Useful for verifying numerical accuracy

```python
distances = backend.compute_distance_matrices(fourier_images, CTF, validate=True)
```

## Troubleshooting

### "MLX not available" Warning

If you see this warning:
```
INFO: MLX not available - using CPU fallback for all operations
```

This means either:
1. MLX is not installed → Install with `pip install mlx`
2. You're not on Apple Silicon → Metal GPU is not available on Intel Macs
3. You're on Linux/Windows → Metal GPU is only available on macOS

### Metal Available but Not Using GPU

If Metal is installed but you don't see the GPU acceleration message:

1. Check the `use_metal_gpu` parameter:
```python
from ManifoldEM.params import params
print(f"use_metal_gpu: {params.use_metal_gpu}")
```

2. Check for explicit disable:
```python
from ManifoldEM.metal_backend import is_metal_available
print(f"Metal available: {is_metal_available()}")
```

### Performance Not Improving

If Metal GPU doesn't improve performance:

- **Small particle counts**: GPU overhead may dominate for <10 particles
- **Small images**: For very small images (<32x32), CPU may be faster
- **Memory pressure**: Check Activity Monitor for memory swapping
- **Thermal throttling**: Sustained computation may thermally throttle the GPU

## Future Work

Potential future enhancements:

- GPU acceleration for sparse eigenvalue decomposition in `DMembeddingII`
- GPU-accelerated image rotation/transformation
- Batch processing optimization for FFT operations
- Support for AMD GPUs on Linux via ROCm
- Support for NVIDIA GPUs via CUDA/CuPy

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/Metal-Performance-Shaders.html)
- [ManifoldEM Documentation](https://github.com/flatironinstitute/ManifoldEM)

## License

The Metal GPU acceleration implementation is part of ManifoldEM and is distributed under the same license (GPLv3).

Copyright (c) 2025 ManifoldEM developers
