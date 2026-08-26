// Package kernel defines shared CUDA launch geometry and memory calculations.
package kernel

import (
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
)

const warpThreads = 32

// NoSharedMemoryBytes returns the launch value for kernels without dynamic shared memory.
func NoSharedMemoryBytes() uint32 { return 0 }

// DefaultThreads returns the standard one-dimensional CUDA block width.
func DefaultThreads() int { return BundleDefaultThreads }

// DefaultBlock1D returns the standard one-dimensional CUDA block geometry.
func DefaultBlock1D() driver.Dim3 {
	return driver.Dim3{X: BundleDefaultThreads, Y: 1, Z: 1}
}

// WarpThreads returns the CUDA warp width.
func WarpThreads() int { return warpThreads }

// WarpBlock1D returns one CUDA warp.
func WarpBlock1D() driver.Dim3 {
	return driver.Dim3{X: warpThreads, Y: 1, Z: 1}
}

// Grid1D returns a one-dimensional CUDA grid containing blocks blocks.
func Grid1D(blocks int) driver.Dim3 {
	return driver.Dim3{X: uint32(blocks), Y: 1, Z: 1}
}

// AttentionSharedMemoryBytes computes bounded dynamic storage for an attention launch.
func AttentionSharedMemoryBytes(spatial, threads int) (uint32, bool) {
	scoreBytes, ok := checked.Mul64(uint64(spatial), binaryschema.Uint32Bytes)
	if !ok {
		return 0, false
	}
	scoreBytes, ok = checked.RoundUpMultiple(scoreBytes, binaryschema.Uint64Bytes)
	if !ok {
		return 0, false
	}
	threadBytes, ok := checked.Mul64(uint64(threads), binaryschema.Uint64Bytes)
	if !ok {
		return 0, false
	}
	total, ok := checked.Add64(scoreBytes, threadBytes)
	return uint32(total), ok && total <= math.MaxUint32
}

// ElementwiseGrid returns the grid required to cover elements with the default block.
func ElementwiseGrid(elements int) driver.Dim3 {
	return driver.Dim3{
		X: uint32((elements + BundleDefaultThreads - 1) / BundleDefaultThreads),
		Y: 1, Z: 1,
	}
}

// VAEConvTile returns the shared convolution tile width used by VAE kernels.
func VAEConvTile() int { return BundleVAEConvTile }
