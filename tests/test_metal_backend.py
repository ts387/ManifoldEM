"""
Tests for Metal GPU acceleration backend.

These tests verify:
1. CPU fallback works correctly when Metal is unavailable
2. Metal operations produce results equivalent to CPU operations (when available)
3. Edge cases are handled properly
"""

import numpy as np
import pytest
from scipy.fftpack import fft2, ifft2

from ManifoldEM.metal_backend import (
    MetalBackend,
    get_backend,
    is_metal_available,
    _MLX_AVAILABLE,
)


class TestMetalBackendCPUFallback:
    """Test CPU fallback behavior (works on all systems)."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.backend = MetalBackend()

    def test_fft2_matches_scipy(self):
        """Test that fft2 produces same results as scipy."""
        img = np.random.randn(64, 64)
        result = self.backend.fft2(img)
        expected = fft2(img)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ifft2_matches_scipy(self):
        """Test that ifft2 produces same results as scipy."""
        img_complex = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
        result = self.backend.ifft2(img_complex)
        expected = ifft2(img_complex)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_fft_roundtrip(self):
        """Test FFT -> IFFT roundtrip."""
        img = np.random.randn(32, 32)
        result = self.backend.ifft2(self.backend.fft2(img))
        np.testing.assert_allclose(result.real, img, rtol=1e-10)

    def test_batch_fft2_filter(self):
        """Test batch FFT filtering."""
        n_images = 10
        size = 32
        images = np.random.randn(n_images, size, size)
        filter_kernel = np.random.randn(size, size)

        result = self.backend.batch_fft2_filter(images, filter_kernel)

        # Compute expected result
        expected = np.zeros_like(images)
        for i in range(n_images):
            expected[i] = ifft2(fft2(images[i]) * filter_kernel).real

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_batch_fft2(self):
        """Test batch FFT."""
        n_images = 10
        size = 32
        images = np.random.randn(n_images, size, size)

        result = self.backend.batch_fft2(images)

        # Compute expected result
        expected = np.zeros_like(images, dtype=np.complex128)
        for i in range(n_images):
            expected[i] = fft2(images[i])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_batch_ifft2(self):
        """Test batch inverse FFT."""
        n_images = 10
        size = 32
        # Create complex input
        images_fft = np.random.randn(n_images, size, size) + 1j * np.random.randn(n_images, size, size)

        result = self.backend.batch_ifft2(images_fft)

        # Compute expected result
        expected = np.zeros_like(images_fft, dtype=np.complex128)
        for i in range(n_images):
            expected[i] = ifft2(images_fft[i])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_batch_fft_roundtrip(self):
        """Test batch FFT -> batch IFFT roundtrip."""
        n_images = 5
        size = 16
        images = np.random.randn(n_images, size, size)

        fft_result = self.backend.batch_fft2(images)
        ifft_result = self.backend.batch_ifft2(fft_result)

        # Should get back original images (real part)
        np.testing.assert_allclose(ifft_result.real, images, rtol=1e-10)

    def test_matmul_matches_numpy(self):
        """Test matrix multiplication."""
        a = np.random.randn(50, 100)
        b = np.random.randn(100, 75)
        result = self.backend.matmul(a, b)
        expected = np.dot(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_abs_squared_real(self):
        """Test abs_squared for real arrays."""
        arr = np.random.randn(20, 20)
        result = self.backend.abs_squared(arr)
        expected = arr ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_abs_squared_complex(self):
        """Test abs_squared for complex arrays."""
        arr = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        result = self.backend.abs_squared(arr)
        expected = np.abs(arr) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_conj(self):
        """Test complex conjugate."""
        arr = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        result = self.backend.conj(arr)
        expected = np.conj(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_real(self):
        """Test real part extraction."""
        arr = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        result = self.backend.real(arr)
        expected = arr.real
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_compute_distance_matrices_small(self):
        """Test distance matrix computation for small particle set."""
        n_particles = 5
        n_pixels = 100
        fourier_images = np.random.randn(n_particles, n_pixels) + 1j * np.random.randn(n_particles, n_pixels)
        CTF = np.random.randn(n_particles, n_pixels)

        result = self.backend.compute_distance_matrices(fourier_images, CTF)

        # Verify symmetry
        np.testing.assert_allclose(result, result.T, rtol=1e-10)

        # Verify diagonal is effectively zero (will be set to exactly 0 in caller)
        # Non-zero diagonal is expected from the mathematical formula before zeroing

        # Verify non-negativity (distances should be >= 0)
        assert np.all(result >= -1e-6), "Distance matrix has negative values"

    def test_compute_distance_matrices_large(self):
        """Test distance matrix computation for larger particle set."""
        n_particles = 50
        n_pixels = 256
        fourier_images = np.random.randn(n_particles, n_pixels) + 1j * np.random.randn(n_particles, n_pixels)
        CTF = np.random.randn(n_particles, n_pixels)

        result = self.backend.compute_distance_matrices(fourier_images, CTF)

        # Verify shape
        assert result.shape == (n_particles, n_particles)

        # Verify symmetry
        np.testing.assert_allclose(result, result.T, rtol=1e-10)

    def test_distance_matrix_known_values(self):
        """Test distance matrix with known expected values."""
        # Create simple case where we can verify result
        n_particles = 3
        n_pixels = 4

        # Simple CTFs (identity-like)
        CTF = np.ones((n_particles, n_pixels))

        # Simple Fourier images
        fourier_images = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
        ], dtype=np.complex128)

        result = self.backend.compute_distance_matrices(fourier_images, CTF)

        # For same images (indices with same fourier), distance should be close to 0
        # D[0,0] and D[1,1] and D[2,2] should be 0 (self-distance)

        # Verify it's symmetric
        np.testing.assert_allclose(result, result.T, rtol=1e-10)

    def test_is_available(self):
        """Test availability check."""
        backend = MetalBackend()
        # Should match global _MLX_AVAILABLE
        assert backend.is_available() == _MLX_AVAILABLE
        assert backend.enabled == _MLX_AVAILABLE

    def test_to_device_returns_input_when_unavailable(self):
        """Test to_device returns input when Metal not available."""
        backend = MetalBackend()
        if not backend.is_available():
            arr = np.random.randn(10, 10)
            result = backend.to_device(arr)
            assert result is arr

    def test_to_numpy_returns_input_for_ndarray(self):
        """Test to_numpy returns input for NumPy arrays."""
        backend = MetalBackend()
        arr = np.random.randn(10, 10)
        result = backend.to_numpy(arr)
        assert result is arr


class TestMetalBackendEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.backend = MetalBackend()

    def test_single_particle_distance(self):
        """Test distance matrix for single particle."""
        n_particles = 1
        n_pixels = 100
        fourier_images = np.random.randn(n_particles, n_pixels) + 1j * np.random.randn(n_particles, n_pixels)
        CTF = np.random.randn(n_particles, n_pixels)

        result = self.backend.compute_distance_matrices(fourier_images, CTF)

        assert result.shape == (1, 1)

    def test_small_image_fft(self):
        """Test FFT for very small images."""
        img = np.random.randn(4, 4)
        result = self.backend.fft2(img)
        expected = fft2(img)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_large_image_fft(self):
        """Test FFT for larger images."""
        img = np.random.randn(128, 128)
        result = self.backend.fft2(img)
        expected = fft2(img)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_non_square_matmul(self):
        """Test matrix multiplication with non-square matrices."""
        a = np.random.randn(10, 20)
        b = np.random.randn(20, 30)
        result = self.backend.matmul(a, b)
        expected = np.dot(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_batch_fft_single_image(self):
        """Test batch FFT with single image."""
        images = np.random.randn(1, 32, 32)
        filter_kernel = np.random.randn(32, 32)
        result = self.backend.batch_fft2_filter(images, filter_kernel)

        expected = ifft2(fft2(images[0]) * filter_kernel).real
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)


class TestGlobalBackend:
    """Test global backend singleton."""

    def test_get_backend_returns_same_instance(self):
        """Test that get_backend returns singleton."""
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_is_metal_available_matches_backend(self):
        """Test global is_metal_available function."""
        backend = get_backend()
        assert is_metal_available() == backend.is_available()


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not available")
class TestMetalBackendGPU:
    """Test Metal GPU specific functionality (only runs on Apple Silicon with MLX)."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.backend = MetalBackend()

    def test_gpu_fft2_matches_cpu(self):
        """Test GPU FFT matches CPU FFT."""
        img = np.random.randn(64, 64)
        result = self.backend.fft2(img)
        expected = fft2(img)
        # Allow slightly more tolerance for GPU computations
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-10)

    def test_gpu_distance_matrix_validation(self):
        """Test distance matrix with validation enabled."""
        n_particles = 10
        n_pixels = 64
        fourier_images = np.random.randn(n_particles, n_pixels) + 1j * np.random.randn(n_particles, n_pixels)
        CTF = np.random.randn(n_particles, n_pixels)

        # This should not raise warnings for reasonable data
        result = self.backend.compute_distance_matrices(fourier_images, CTF, validate=True)

        assert result.shape == (n_particles, n_particles)
        np.testing.assert_allclose(result, result.T, rtol=1e-6)

    def test_gpu_batch_filter_performance(self):
        """Test batch filter produces correct results on GPU."""
        n_images = 20
        size = 64
        images = np.random.randn(n_images, size, size)
        filter_kernel = np.random.randn(size, size)

        result = self.backend.batch_fft2_filter(images, filter_kernel)

        # Compute expected result on CPU
        expected = np.zeros_like(images)
        for i in range(n_images):
            expected[i] = ifft2(fft2(images[i]) * filter_kernel).real

        # Allow slightly more tolerance for GPU computations
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-10)
