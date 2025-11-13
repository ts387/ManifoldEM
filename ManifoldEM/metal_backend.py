"""
Metal GPU acceleration backend for ManifoldEM using Apple's MLX framework.

This module provides GPU-accelerated computational operations for Mac systems with Apple Silicon,
with automatic fallback to CPU operations when Metal/MLX is not available.

Copyright (c) 2025 ManifoldEM developers
"""

import logging
import platform
import numpy as np
from typing import Optional, Tuple, Any

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

# Try to import MLX for Metal GPU acceleration
_MLX_AVAILABLE = False
_METAL_DEVICE = None

try:
    import mlx.core as mx
    import mlx.core.fft as mx_fft

    # Check if we're on macOS with Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        _MLX_AVAILABLE = True
        _METAL_DEVICE = mx.gpu
        _logger.info("MLX Metal GPU acceleration enabled on Apple Silicon")
    else:
        _logger.info("MLX found but not on Apple Silicon - using CPU fallback")
except ImportError:
    _logger.info("MLX not available - using CPU fallback for all operations")


class MetalBackend:
    """
    Metal GPU acceleration backend with automatic fallback to CPU.

    Provides drop-in replacements for NumPy/SciPy operations that can be
    accelerated on Apple Silicon GPUs via MLX.
    """

    def __init__(self):
        self.enabled = _MLX_AVAILABLE
        self.device = _METAL_DEVICE if _MLX_AVAILABLE else None

    def is_available(self) -> bool:
        """Check if Metal GPU acceleration is available."""
        return self.enabled

    def to_device(self, array: np.ndarray) -> Any:
        """
        Transfer NumPy array to Metal GPU device.

        Parameters
        ----------
        array : np.ndarray
            Input array to transfer

        Returns
        -------
        mlx.core.array or np.ndarray
            Array on GPU if Metal is available, otherwise original array
        """
        if not self.enabled:
            return array
        return mx.array(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        """
        Transfer array from Metal GPU back to NumPy.

        Parameters
        ----------
        array : mlx.core.array or np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            NumPy array on CPU
        """
        if not self.enabled or isinstance(array, np.ndarray):
            return array
        return np.array(array)

    def fft2(self, array: np.ndarray) -> np.ndarray:
        """
        2D Fast Fourier Transform with Metal GPU acceleration.

        Parameters
        ----------
        array : np.ndarray
            Input 2D array (real or complex)

        Returns
        -------
        np.ndarray
            2D FFT of input array
        """
        if not self.enabled:
            from scipy.fftpack import fft2
            return fft2(array)

        # Transfer to GPU, compute FFT, transfer back
        gpu_array = mx.array(array)
        result = mx_fft.fft2(gpu_array)
        return np.array(result)

    def ifft2(self, array: np.ndarray) -> np.ndarray:
        """
        2D Inverse Fast Fourier Transform with Metal GPU acceleration.

        Parameters
        ----------
        array : np.ndarray
            Input 2D array (complex)

        Returns
        -------
        np.ndarray
            2D inverse FFT of input array
        """
        if not self.enabled:
            from scipy.fftpack import ifft2
            return ifft2(array)

        # Transfer to GPU, compute inverse FFT, transfer back
        gpu_array = mx.array(array)
        result = mx_fft.ifft2(gpu_array)
        return np.array(result)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with Metal GPU acceleration.

        Parameters
        ----------
        a : np.ndarray
            First input array
        b : np.ndarray
            Second input array

        Returns
        -------
        np.ndarray
            Matrix product of a and b
        """
        if not self.enabled:
            return np.dot(a, b)

        # Transfer to GPU, compute matmul, transfer back
        gpu_a = mx.array(a)
        gpu_b = mx.array(b)
        result = mx.matmul(gpu_a, gpu_b)
        return np.array(result)

    def abs_squared(self, array: np.ndarray) -> np.ndarray:
        """
        Compute absolute value squared with Metal GPU acceleration.

        For complex arrays, this computes |z|^2 = real^2 + imag^2

        Parameters
        ----------
        array : np.ndarray
            Input array (real or complex)

        Returns
        -------
        np.ndarray
            Absolute value squared of input
        """
        if not self.enabled:
            return np.abs(array) ** 2

        gpu_array = mx.array(array)
        result = mx.abs(gpu_array) ** 2
        return np.array(result)

    def conj(self, array: np.ndarray) -> np.ndarray:
        """
        Complex conjugate with Metal GPU acceleration.

        Parameters
        ----------
        array : np.ndarray
            Input complex array

        Returns
        -------
        np.ndarray
            Complex conjugate of input
        """
        if not self.enabled:
            return np.conj(array)

        gpu_array = mx.array(array)
        result = mx.conjugate(gpu_array)
        return np.array(result)

    def real(self, array: np.ndarray) -> np.ndarray:
        """
        Extract real part with Metal GPU acceleration.

        Parameters
        ----------
        array : np.ndarray
            Input complex array

        Returns
        -------
        np.ndarray
            Real part of input
        """
        if not self.enabled:
            return array.real

        gpu_array = mx.array(array)
        result = mx.real(gpu_array)
        return np.array(result)

    def compute_distance_matrices(
        self,
        fourier_images: np.ndarray,
        CTF: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance matrices for image comparison with Metal GPU acceleration.

        This is the main computational bottleneck in calc_distance.py.
        Computes D[i,j] = sum(|CTF[i] * fourier_images[j] - CTF[j] * fourier_images[i]|^2)

        Parameters
        ----------
        fourier_images : np.ndarray
            Fourier transforms of images, shape (n_particles, n_pixels^2)
        CTF : np.ndarray
            Contrast transfer functions, shape (n_particles, n_pixels^2)

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_particles, n_particles)
        """
        if not self.enabled:
            # CPU fallback - original implementation
            CTFfy = CTF.conj() * fourier_images
            distances = np.dot((np.abs(CTF) ** 2), (np.abs(fourier_images) ** 2).T)
            distances = (
                distances + distances.T - 2 * np.real(np.dot(CTFfy, CTFfy.conj().transpose()))
            )
            return distances

        # GPU-accelerated version using MLX
        _logger.debug("Computing distance matrices on Metal GPU")

        # Transfer data to GPU
        gpu_fourier = mx.array(fourier_images)
        gpu_ctf = mx.array(CTF)

        # Compute CTF * fourier_images (element-wise)
        CTFfy = mx.conjugate(gpu_ctf) * gpu_fourier

        # Compute |CTF|^2
        abs_ctf_sq = mx.abs(gpu_ctf) ** 2

        # Compute |fourier_images|^2
        abs_fourier_sq = mx.abs(gpu_fourier) ** 2

        # Matrix multiplication: |CTF|^2 @ |fourier|^2.T
        distances = mx.matmul(abs_ctf_sq, abs_fourier_sq.T)

        # Make symmetric and subtract cross term
        distances = distances + distances.T - 2 * mx.real(
            mx.matmul(CTFfy, mx.conjugate(CTFfy).T)
        )

        # Transfer back to CPU as NumPy array
        return np.array(distances)


# Global singleton instance
_backend = MetalBackend()


def get_backend() -> MetalBackend:
    """
    Get the global Metal backend instance.

    Returns
    -------
    MetalBackend
        The global backend instance
    """
    return _backend


def is_metal_available() -> bool:
    """
    Check if Metal GPU acceleration is available.

    Returns
    -------
    bool
        True if Metal/MLX is available and enabled
    """
    return _backend.is_available()


def enable_metal() -> bool:
    """
    Attempt to enable Metal GPU acceleration.

    Returns
    -------
    bool
        True if Metal was successfully enabled
    """
    if _MLX_AVAILABLE:
        _backend.enabled = True
        _logger.info("Metal GPU acceleration enabled")
        return True
    else:
        _logger.warning("Cannot enable Metal: MLX not available")
        return False


def disable_metal():
    """Disable Metal GPU acceleration and use CPU fallback."""
    _backend.enabled = False
    _logger.info("Metal GPU acceleration disabled, using CPU fallback")
