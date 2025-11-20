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


def _ensure_mlx_imported():
    """Ensure MLX modules are imported. Returns (mx, mx_fft) or raises ImportError."""
    if not _MLX_AVAILABLE:
        raise ImportError("MLX is not available")
    import mlx.core as mx
    import mlx.core.fft as mx_fft
    return mx, mx_fft


class MetalBackend:
    """
    Metal GPU acceleration backend with automatic fallback to CPU.

    Provides drop-in replacements for NumPy/SciPy operations that can be
    accelerated on Apple Silicon GPUs via MLX.

    Note: This class is designed to be thread-safe. The `enabled` property
    should not be mutated during parallel execution.
    """

    def __init__(self):
        self._mlx_available = _MLX_AVAILABLE
        self.device = _METAL_DEVICE if _MLX_AVAILABLE else None

    @property
    def enabled(self) -> bool:
        """Check if Metal GPU acceleration is currently enabled."""
        return self._mlx_available

    def is_available(self) -> bool:
        """Check if Metal GPU acceleration is available (alias for enabled)."""
        return self._mlx_available

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
        if not self._mlx_available:
            return array
        mx, _ = _ensure_mlx_imported()
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
        if not self._mlx_available or isinstance(array, np.ndarray):
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
        if not self._mlx_available:
            from scipy.fftpack import fft2
            return fft2(array)

        mx, mx_fft = _ensure_mlx_imported()
        gpu_array = mx.array(array)
        result = mx_fft.fft2(gpu_array)
        mx.eval(result)  # Force evaluation
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
        if not self._mlx_available:
            from scipy.fftpack import ifft2
            return ifft2(array)

        mx, mx_fft = _ensure_mlx_imported()
        gpu_array = mx.array(array)
        result = mx_fft.ifft2(gpu_array)
        mx.eval(result)  # Force evaluation
        return np.array(result)

    def batch_fft2_filter(self, images: np.ndarray, filter_kernel: np.ndarray) -> np.ndarray:
        """
        Batch 2D FFT filtering with Metal GPU acceleration.

        Applies FFT, multiplies by filter kernel, and applies inverse FFT for all images at once.
        This is more efficient than individual operations due to reduced GPU transfer overhead.

        Parameters
        ----------
        images : np.ndarray
            Input 3D array of images, shape (n_images, height, width)
        filter_kernel : np.ndarray
            2D filter kernel in frequency domain, shape (height, width)

        Returns
        -------
        np.ndarray
            Filtered images (real part), shape (n_images, height, width)
        """
        if not self._mlx_available:
            from scipy.fftpack import fft2, ifft2
            result = np.zeros_like(images, dtype=np.float64)
            for i in range(images.shape[0]):
                result[i] = ifft2(fft2(images[i]) * filter_kernel).real
            return result

        mx, mx_fft = _ensure_mlx_imported()

        n_images = images.shape[0]
        result = np.zeros_like(images, dtype=np.float64)

        # Transfer filter once
        gpu_filter = mx.array(filter_kernel)

        # Process images individually (MLX fft2 doesn't support 3D batch processing)
        # But we minimize transfers by keeping intermediate results on GPU
        for i in range(n_images):
            gpu_img = mx.array(images[i])
            fft_img = mx_fft.fft2(gpu_img)
            filtered = fft_img * gpu_filter
            ifft_result = mx_fft.ifft2(filtered)
            result_real = mx.real(ifft_result)
            mx.eval(result_real)
            result[i] = np.array(result_real)

        return result

    def batch_fft2(self, images: np.ndarray) -> np.ndarray:
        """
        Batch 2D FFT with Metal GPU acceleration.

        Computes FFT for multiple images efficiently.

        Parameters
        ----------
        images : np.ndarray
            Input 3D array of images, shape (n_images, height, width)

        Returns
        -------
        np.ndarray
            Fourier transforms, shape (n_images, height, width), complex
        """
        if not self._mlx_available:
            from scipy.fftpack import fft2
            result = np.zeros_like(images, dtype=np.complex128)
            for i in range(images.shape[0]):
                result[i] = fft2(images[i])
            return result

        mx, mx_fft = _ensure_mlx_imported()

        n_images = images.shape[0]
        result = np.zeros_like(images, dtype=np.complex128)

        # Process each image (avoiding multiple Python-to-GPU transfers in loop)
        for i in range(n_images):
            gpu_img = mx.array(images[i])
            fft_result = mx_fft.fft2(gpu_img)
            mx.eval(fft_result)
            result[i] = np.array(fft_result)

        return result

    def batch_ifft2(self, images_fft: np.ndarray) -> np.ndarray:
        """
        Batch 2D inverse FFT with Metal GPU acceleration.

        Computes inverse FFT for multiple images efficiently.

        Parameters
        ----------
        images_fft : np.ndarray
            Input 3D array of Fourier transforms, shape (n_images, height, width), complex

        Returns
        -------
        np.ndarray
            Inverse Fourier transforms, shape (n_images, height, width), complex
        """
        if not self._mlx_available:
            from scipy.fftpack import ifft2
            result = np.zeros_like(images_fft, dtype=np.complex128)
            for i in range(images_fft.shape[0]):
                result[i] = ifft2(images_fft[i])
            return result

        mx, mx_fft = _ensure_mlx_imported()

        n_images = images_fft.shape[0]
        result = np.zeros_like(images_fft, dtype=np.complex128)

        # Process each image
        for i in range(n_images):
            gpu_img = mx.array(images_fft[i])
            ifft_result = mx_fft.ifft2(gpu_img)
            mx.eval(ifft_result)
            result[i] = np.array(ifft_result)

        return result

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
        if not self._mlx_available:
            return np.dot(a, b)

        mx, _ = _ensure_mlx_imported()
        gpu_a = mx.array(a)
        gpu_b = mx.array(b)
        result = mx.matmul(gpu_a, gpu_b)
        mx.eval(result)
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
        if not self._mlx_available:
            return np.abs(array) ** 2

        mx, _ = _ensure_mlx_imported()
        gpu_array = mx.array(array)
        result = mx.abs(gpu_array) ** 2
        mx.eval(result)
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
        if not self._mlx_available:
            return np.conj(array)

        mx, _ = _ensure_mlx_imported()
        gpu_array = mx.array(array)
        result = mx.conjugate(gpu_array)
        mx.eval(result)
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
        if not self._mlx_available:
            return array.real

        mx, _ = _ensure_mlx_imported()
        gpu_array = mx.array(array)
        result = mx.real(gpu_array)
        mx.eval(result)
        return np.array(result)

    def compute_distance_matrices(
        self,
        fourier_images: np.ndarray,
        CTF: np.ndarray,
        validate: bool = False
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
        validate : bool, default=False
            If True, compare GPU result with CPU result and warn if difference is large

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_particles, n_particles)
        """
        # CPU fallback implementation (also used for validation)
        def cpu_compute():
            CTFfy = CTF.conj() * fourier_images
            distances = np.dot((np.abs(CTF) ** 2), (np.abs(fourier_images) ** 2).T)
            distances = (
                distances + distances.T - 2 * np.real(np.dot(CTFfy, CTFfy.conj().transpose()))
            )
            return distances

        if not self._mlx_available:
            return cpu_compute()

        mx, _ = _ensure_mlx_imported()

        # GPU-accelerated version using MLX
        _logger.debug("Computing distance matrices on Metal GPU")

        # Transfer data to GPU (single transfer for efficiency)
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

        # Force evaluation and transfer back to CPU
        mx.eval(distances)
        result = np.array(distances)

        # Optional validation against CPU result
        if validate:
            cpu_result = cpu_compute()
            max_diff = np.max(np.abs(result - cpu_result))
            rel_diff = max_diff / (np.max(np.abs(cpu_result)) + 1e-10)
            if rel_diff > 1e-6:
                _logger.warning(
                    f"Metal GPU distance computation differs from CPU by {rel_diff:.2e} (max abs diff: {max_diff:.2e})"
                )
            else:
                _logger.debug(f"Metal GPU validation passed (rel diff: {rel_diff:.2e})")

        return result


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

    Note
    ----
    This function is deprecated. Metal is automatically enabled if available.
    """
    if _MLX_AVAILABLE:
        _logger.info("Metal GPU acceleration is available")
        return True
    else:
        _logger.warning("Cannot enable Metal: MLX not available")
        return False


def disable_metal():
    """
    Disable Metal GPU acceleration and use CPU fallback.

    Note
    ----
    This function is deprecated. Use params.use_metal_gpu = False instead.
    """
    _logger.warning("disable_metal() is deprecated. Use params.use_metal_gpu = False instead.")
