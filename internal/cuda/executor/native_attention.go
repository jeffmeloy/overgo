package executor

import (
	"errors"
	"math"
	"math/bits"

	"overgo/internal/checked"
	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// Native attention staging follows Q, K, V, scores and output in the CUDA ABI.
// Reconsider these slots with attention_native_pack_f32 when changing reuse.
const (
	nativeAttentionQuery = iota
	nativeAttentionKey
	nativeAttentionValue
	nativeAttentionScores
	nativeAttentionOutput
	nativeAttentionBuffers
)

// Native softmax uses CUDA warp shuffles from Torch PersistentSoftmax.cuh.
const nativeAttentionWarp = 32

// Native softmax follows Torch SoftMax.cu's persistent threshold and block reduction size.
// Reconsider these values with a different source reduction implementation.
const (
	nativeAttentionPersistentWidth = 2048
	nativeAttentionBlockThreads    = 1024
)

type nativeAttentionShape struct {
	width, valueWidth, heads, keyHeads, queries, keys, batches, groups uint32
	sizes                                                              [nativeAttentionBuffers]uint64
	bytes                                                              uint64
}

func nativeAttentionGeometry(node *tensor.Tensor) (nativeAttentionShape, error) {
	var g nativeAttentionShape
	a := node.Attrs.(tensor.AttentionAttributes)
	if a.KeyValueTokens != 0 || a.HasKeyBias || a.HasSinks || a.HasBlockMask || a.QueryStart != 0 ||
		a.Window != 0 || a.Softcap != 0 || a.RelativeBuckets != 0 || a.MaxALiBiBias != 0 ||
		a.SymmetricWindow || a.ChunkedWindow || a.NaiveF32 || a.MemoryEfficientF32 {
		return g, errors.New("native math attention requires static unpadded keys and only an optional causal mask")
	}
	q, k, v := node.Inputs[0].Shape, node.Inputs[1].Shape, node.Inputs[tensor.PairedExtent].Shape
	batch := uint64(1)
	if q.Rank == tensor.MaxDimensions {
		batch = q.Dims[tensor.TripleExtent]
	}
	dims := []uint64{q.Dims[0], v.Dims[0], q.Dims[1], k.Dims[1], q.Dims[tensor.PairedExtent], k.Dims[tensor.PairedExtent], batch}
	for _, dim := range dims {
		if dim == 0 || dim > math.MaxInt32 {
			return g, errors.New("native math attention dimensions exceed positive cuBLAS limits")
		}
	}
	if k.Dims[tensor.PairedExtent] > tensor.NativeAttentionSoftmaxWidth || batch > math.MaxInt32/q.Dims[1] {
		return g, errors.New("native math attention key count or head batch exceeds source limits")
	}
	g.width, g.valueWidth, g.heads, g.keyHeads = uint32(q.Dims[0]), uint32(v.Dims[0]), uint32(q.Dims[1]), uint32(k.Dims[1])
	g.queries, g.keys, g.batches = uint32(q.Dims[tensor.PairedExtent]), uint32(k.Dims[tensor.PairedExtent]), uint32(batch)
	g.groups = g.heads * g.batches
	g.sizes = [nativeAttentionBuffers]uint64{uint64(g.queries) * uint64(g.width), uint64(g.keys) * uint64(g.width), uint64(g.keys) * uint64(g.valueWidth), uint64(g.queries) * uint64(g.keys), uint64(g.queries) * uint64(g.valueWidth)}
	for _, size := range g.sizes {
		count, ok := checked.Mul64(size, uint64(g.groups))
		if !ok || count > math.MaxUint32 {
			return g, errors.New("native math attention launch elements exceed uint32")
		}
		bytes, ok := checked.Mul64(count, f32ScalarBytes)
		if !ok {
			return g, errors.New("native math attention staging bytes overflow")
		}
		var err error
		g.bytes, err = addMemoryBytes(g.bytes, bytes)
		if err != nil {
			return g, err
		}
	}
	return g, nil
}

func launchNativeAttention(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, pointers launchPointerFrame) error {
	g, err := nativeAttentionGeometry(node)
	if err != nil {
		return err
	}
	if blas == nil || blas.handle == 0 || blas.stagingBytes < g.bytes {
		return errors.New("native math attention requires prepared BLAS staging")
	}
	var buffers [nativeAttentionBuffers]driver.DevicePtr
	next := blas.staging
	for i, count := range g.sizes {
		buffers[i] = next
		next += driver.DevicePtr(count * uint64(g.groups) * f32ScalarBytes)
	}
	blas.stagedNode = nil
	query, key, value, output := pointers.input(0), pointers.input(1), pointers.input(tensor.PairedExtent), pointers.output()
	a := node.Attrs.(tensor.AttentionAttributes)
	scale := float32(math.Sqrt(float64(a.Scale)))
	pack := func(phase uint32, count uint64) error {
		n := uint32(count * uint64(g.groups))
		return launch1DABI(state, functions[kernelAttentionNativePackF32], n,
			&query, &key, &value, &output, &buffers[nativeAttentionQuery], &buffers[nativeAttentionKey], &buffers[nativeAttentionValue], &buffers[nativeAttentionOutput],
			&g.width, &g.valueWidth, &g.heads, &g.keyHeads, &g.queries, &g.keys, &g.groups, &scale, &phase, &n)
	}
	if err = pack(0, max(g.sizes[nativeAttentionQuery], g.sizes[nativeAttentionKey], g.sizes[nativeAttentionValue])); err != nil {
		return err
	}
	product := func(left, right, destination int, transpose cublas.Operation, m, n, k, ldLeft, ldRight, ldOut uint32) error {
		if traceExternalCall(state, traceTagSGEMMStrided, uint64(buffers[left]), uint64(buffers[right]), uint64(buffers[destination]), uint64(m), uint64(n), uint64(k), uint64(g.groups), uint64(transpose)) {
			return nil
		}
		return blas.library.SGEMMStridedBatched(blas.handle, transpose, cublas.OperationNone, int32(m), int32(n), int32(k), 1, buffers[left], int32(ldLeft), int64(g.sizes[left]), buffers[right], int32(ldRight), int64(g.sizes[right]), 0, buffers[destination], int32(ldOut), int64(g.sizes[destination]), int32(g.groups))
	}
	if err = product(nativeAttentionKey, nativeAttentionQuery, nativeAttentionScores, cublas.OperationTranspose, g.keys, g.queries, g.width, g.width, g.width, g.keys); err != nil {
		return err
	}
	rows, causal := g.groups*g.queries, kernelBool(a.Causal)
	lanes := min(uint32(nativeAttentionWarp), uint32(1)<<bits.Len32(g.keys-1))
	if g.keys > nativeAttentionPersistentWidth {
		lanes = nativeAttentionBlockThreads
	}
	if err = launchGridABI(state, functions[kernelAttentionNativeSoftmaxF32], driver.Dim3{X: rows, Y: 1, Z: 1}, driver.Dim3{X: lanes, Y: 1, Z: 1}, &buffers[nativeAttentionScores], &g.keys, &g.queries, &rows, &causal); err != nil {
		return err
	}
	if err = product(nativeAttentionValue, nativeAttentionScores, nativeAttentionOutput, cublas.OperationNone, g.valueWidth, g.queries, g.keys, g.valueWidth, g.keys, g.valueWidth); err != nil {
		return err
	}
	return pack(1, g.sizes[nativeAttentionOutput])
}
