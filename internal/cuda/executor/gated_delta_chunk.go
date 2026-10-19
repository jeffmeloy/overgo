package executor

import (
	"errors"
	"math"
	"math/bits"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
)

// The chunk workspace ABI holds nine token matrices, two attention matrices,
// one scan row and two state matrices per head from the native chunk equations.
// Reconsider these slots together with gated_delta_chunk_f32 when changing reuse.
const (
	deltaQuery = iota
	deltaKey
	deltaBetaKey
	deltaBetaValue
	deltaValue
	deltaCumulativeKey
	deltaNewValue
	deltaExpQuery
	deltaTemporary
	deltaTriangle
	deltaScores
	deltaCumulativeGate
	deltaState
	deltaStateUpdate
	deltaBufferCount
)

// Chunk phases follow the data dependencies between native cuBLAS products.
// Reconsider their ordering with the kernel's phase dispatch.
const (
	deltaPrepare = iota
	deltaResolveTriangle
	deltaWeightInputs
	deltaResidual
	deltaWriteAttention
	deltaWriteState
)

// deltaScanThreadBits follows ScanUtils.cuh's 512-thread budget and 16 x-lane minimum.
// Reconsider both exponents when changing the native CUDA scan contract.
const (
	deltaScanThreadBits  = 9
	deltaScanMinimumBits = 4
)

func deltaChunkBufferElements(width uint64) [deltaBufferCount]uint64 {
	var sizes [deltaBufferCount]uint64
	for i := range sizes {
		switch {
		case i < deltaTriangle:
			sizes[i] = tensor.GatedDeltaChunkSize * width
		case i < deltaCumulativeGate:
			sizes[i] = tensor.GatedDeltaChunkSize * tensor.GatedDeltaChunkSize
		case i == deltaCumulativeGate:
			sizes[i] = tensor.GatedDeltaChunkSize
		default:
			sizes[i] = width * width
		}
	}
	return sizes
}

func deltaChunkStagingBytes(node *tensor.Tensor) (uint64, error) {
	if node.Inputs[tensor.TripleExtent].Shape.Dims[0] != tensor.SingletonExtent {
		return 0, errors.New("chunked GatedDeltaNet requires scalar gates")
	}
	dims := node.Inputs[tensor.PairedExtent].Shape.Dims
	width, heads, tokens, sequences := dims[0], dims[1], dims[tensor.PairedExtent], dims[tensor.TripleExtent]
	if width == 0 || width > math.MaxInt32 || heads == 0 || tokens == 0 || tokens > math.MaxUint32 || sequences == 0 || heads > math.MaxInt32/sequences {
		return 0, errors.New("chunked GatedDeltaNet dimensions exceed positive CUDA and cuBLAS limits")
	}
	var elements uint64
	for _, count := range deltaChunkBufferElements(width) {
		var err error
		elements, err = addMemoryBytes(elements, count)
		if err != nil {
			return 0, err
		}
	}
	elements, ok := checked.Mul64(elements, heads*sequences)
	if !ok {
		return 0, errors.New("chunked GatedDeltaNet staging element count overflows")
	}
	bytes, ok := checked.Mul64(elements, f32ScalarBytes)
	if !ok {
		return 0, errors.New("chunked GatedDeltaNet staging byte count overflows")
	}
	return bytes, nil
}

func launchDeltaChunk(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, pointers launchPointerFrame) error {
	bytes, err := deltaChunkStagingBytes(node)
	if err != nil {
		return err
	}
	if blas == nil || blas.handle == 0 || blas.stagingBytes < bytes {
		return errors.New("chunked GatedDeltaNet requires prepared BLAS staging")
	}
	dims := shapeDimensions32(node.Inputs[tensor.PairedExtent].Shape)
	width, heads, tokens, sequences := dims[0], dims[1], dims[tensor.PairedExtent], dims[tensor.TripleExtent]
	groups := heads * sequences
	qHeads, kHeads := uint32(node.Inputs[0].Shape.Dims[1]), uint32(node.Inputs[1].Shape.Dims[1])
	query, key, value, gate, beta, initial := pointers.input(0), pointers.input(1), pointers.input(tensor.PairedExtent), pointers.input(tensor.TripleExtent), pointers.input(tensor.TripleExtent+1), pointers.input(tensor.TripleExtent+tensor.PairedExtent)
	output, workspace := pointers.output(), blas.staging
	var buffers [deltaBufferCount]driver.DevicePtr
	next := workspace
	for i, count := range deltaChunkBufferElements(uint64(width)) {
		buffers[i] = next
		next += driver.DevicePtr(count * uint64(groups) * f32ScalarBytes)
	}
	chunks := (uint64(tokens) + tensor.GatedDeltaChunkSize - 1) / tensor.GatedDeltaChunkSize
	// ScanUtils.cuh balances 512 threads across scan rows with at least 16 x lanes.
	// The chunk row always has 64 elements and may use one or two scan blocks.
	logRows := bits.Len64(uint64(groups)*chunks - 1)
	logWidth := bits.Len64(tensor.GatedDeltaChunkSize - 1)
	logX := max(deltaScanMinimumBits, min(deltaScanThreadBits, (deltaScanThreadBits+logWidth-logRows)/tensor.PairedExtent))
	scanBlock := min(uint32(tensor.GatedDeltaChunkSize), uint32(1<<(logX+1)))
	scale := float32(1 / math.Sqrt(float64(width)))
	repeat := kernelBool(node.Attrs.(tensor.GatedDeltaNetAttributes).RepeatInterleave)
	blas.stagedNode = nil
	launch := func(start, phase uint32) error {
		return launchGridABI(state, functions[kernelGatedDeltaChunkF32], driver.Dim3{X: groups, Y: 1, Z: 1}, kernel.DefaultBlock1D(),
			&query, &key, &value, &gate, &beta, &initial, &output, &workspace,
			&width, &qHeads, &kHeads, &heads, &tokens, &sequences, &start, &scanBlock, &repeat, &phase, &scale)
	}
	product := func(a, b, c int, m, k, n uint32, transposeA, transposeB bool) error {
		if traceExternalCall(state, traceTagSGEMMStrided, uint64(buffers[a]), uint64(buffers[b]), uint64(buffers[c]), uint64(m), uint64(k), uint64(n), uint64(groups), uint64(kernelBool(transposeA)), uint64(kernelBool(transposeB))) {
			return nil
		}
		return blas.library.RowMajorGEMMStridedBatchedF32(blas.handle, transposeA, transposeB, int32(m), int32(k), int32(n), buffers[a], int64(m)*int64(k), buffers[b], int64(k)*int64(n), buffers[c], int64(m)*int64(n), int32(groups))
	}
	for chunk := range chunks {
		start := uint32(chunk * tensor.GatedDeltaChunkSize)
		if err = launch(start, deltaPrepare); err != nil {
			return err
		}
		if err = product(deltaBetaKey, deltaKey, deltaTriangle, tensor.GatedDeltaChunkSize, width, tensor.GatedDeltaChunkSize, false, true); err != nil {
			return err
		}
		if err = launch(start, deltaResolveTriangle); err != nil {
			return err
		}
		if err = launch(start, deltaWeightInputs); err != nil {
			return err
		}
		if err = product(deltaTriangle, deltaBetaValue, deltaValue, tensor.GatedDeltaChunkSize, tensor.GatedDeltaChunkSize, width, false, false); err != nil {
			return err
		}
		if err = product(deltaTriangle, deltaBetaKey, deltaCumulativeKey, tensor.GatedDeltaChunkSize, tensor.GatedDeltaChunkSize, width, false, false); err != nil {
			return err
		}
		if err = product(deltaQuery, deltaKey, deltaScores, tensor.GatedDeltaChunkSize, width, tensor.GatedDeltaChunkSize, false, true); err != nil {
			return err
		}
		if err = product(deltaCumulativeKey, deltaState, deltaTemporary, tensor.GatedDeltaChunkSize, width, width, false, false); err != nil {
			return err
		}
		if err = launch(start, deltaResidual); err != nil {
			return err
		}
		if err = product(deltaExpQuery, deltaState, deltaTemporary, tensor.GatedDeltaChunkSize, width, width, false, false); err != nil {
			return err
		}
		if err = product(deltaScores, deltaNewValue, deltaBetaValue, tensor.GatedDeltaChunkSize, tensor.GatedDeltaChunkSize, width, false, false); err != nil {
			return err
		}
		if err = launch(start, deltaWriteAttention); err != nil {
			return err
		}
		if err = product(deltaBetaKey, deltaNewValue, deltaStateUpdate, width, tensor.GatedDeltaChunkSize, width, true, false); err != nil {
			return err
		}
		if err = launch(start, deltaWriteState); err != nil {
			return err
		}
	}
	return nil
}
