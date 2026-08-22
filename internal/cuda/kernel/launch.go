package kernel

import (
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
)

const (
	defaultThreads = 256
	vaeConvTile    = 64
)

func NoSharedMemoryBytes() uint32 { return 0 }

func DefaultThreads() int { return defaultThreads }

func DefaultBlock1D() driver.Dim3 {
	return driver.Dim3{X: defaultThreads, Y: 1, Z: 1}
}

func Grid1D(blocks int) driver.Dim3 {
	return driver.Dim3{X: uint32(blocks), Y: 1, Z: 1}
}

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

func ElementwiseGrid(elements int) driver.Dim3 {
	return driver.Dim3{X: uint32((elements + defaultThreads - 1) / defaultThreads), Y: 1, Z: 1}
}

func VAEConvTile() int { return vaeConvTile }
