//go:build windows

package devicemath

import (
	"context"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostmath"
)

// frozenSlabAlignment is the floats each buffer carved from a frozen layer's one allocation starts on a multiple of:
// 256 bytes, cuMemAlloc's alignment, so a carved buffer is read as one allocated alone would be.
const frozenSlabAlignment = 64

// frozenBuffer is one buffer carved from a frozen layer's allocation: where its pointer goes and the values it holds.
type frozenBuffer struct {
	pointer *driver.DevicePtr
	count   int
}

// frozenWeight is a frozen layer's buffer that holds a weight, data put on the card when the layer opens.
type frozenWeight struct {
	frozenBuffer
	data []float32
}

// frozenMix is the token mix a frozen decoder layer wraps. forwardMix reads the input norm's output from the layer's
// normed rows and writes the mix's output to its mixed rows; backwardMix reads the gradient with respect to that output
// from mixed, which it leaves as it found it, and writes the gradient with respect to the normed input to normed.
type frozenMix interface {
	forwardMix(r *frozenRun)
	backwardMix(r *frozenRun)
}

// frozenLayer is what every frozen decoder layer on the card holds around its mix, as hostmath.HybridDecoderLayerForward
// composes it: out = h + MLP(RMSNorm(h, PostNorm)), h = x + Mix(RMSNorm(x, InputNorm)), the MLP SwiGLU. Its weights
// are put on the card once in F32, in one allocation with its cache and scratch; Forward keeps on the card what
// Backward reads, and Backward gives the gradient with respect to the layer's input alone.
type frozenLayer struct {
	session           *ResidentOpsSession
	name              string
	mix               frozenMix
	d                 hostmath.HybridLayerDims
	slab              driver.DevicePtr
	around            frozenAround
	forwarded, closed bool
}

// frozenAround is what the pre-norms, the MLP and the residuals around a mix hold on the card: the norms' and the MLP's
// weights; the forward's input, mid residual h, MLP gate and up products and output, which the backward reads; and
// scratch, hidden rows (normed and mixed, as frozenMix reads them, and the input gradient) and MLP rows (two).
type frozenAround struct {
	inputNorm, postNorm, gate, up, down driver.DevicePtr
	x, h, mlpGate, mlpUp, out           driver.DevicePtr
	normed, mixed, dx, inter, inter2    driver.DevicePtr
}

// open puts w's norms and MLP and the mix's weights on the card, carving the layer's one allocation into them, the
// buffers around the mix and the mix's own, which the layer releases on Close; on a failure it releases what it took.
func (l *frozenLayer) open(s *ResidentOpsSession, name string, mix frozenMix, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, weights []frozenWeight, buffers []frozenBuffer) error {
	l.session, l.name, l.mix, l.d = s, name, mix, d
	tokens, hidden, inter, a := d.Tokens, d.Hidden, d.Inter, &l.around
	weights = append(weights,
		frozenWeight{frozenBuffer{&a.inputNorm, hidden}, w.InputNorm}, frozenWeight{frozenBuffer{&a.postNorm, hidden}, w.PostNorm},
		frozenWeight{frozenBuffer{&a.gate, inter * hidden}, w.MLP.Gate}, frozenWeight{frozenBuffer{&a.up, inter * hidden}, w.MLP.Up},
		frozenWeight{frozenBuffer{&a.down, hidden * inter}, w.MLP.Down})
	buffers = append(buffers,
		frozenBuffer{&a.x, tokens * hidden}, frozenBuffer{&a.h, tokens * hidden}, frozenBuffer{&a.mlpGate, tokens * inter},
		frozenBuffer{&a.mlpUp, tokens * inter}, frozenBuffer{&a.out, tokens * hidden}, frozenBuffer{&a.normed, tokens * hidden},
		frozenBuffer{&a.mixed, tokens * hidden}, frozenBuffer{&a.dx, tokens * hidden}, frozenBuffer{&a.inter, tokens * inter},
		frozenBuffer{&a.inter2, tokens * inter})
	for index, weight := range weights {
		if len(weight.data) != weight.count {
			return fmt.Errorf("%s: weight %d holds %d values, want %d", name, index, len(weight.data), weight.count)
		}
		buffers = append(buffers, weight.frozenBuffer)
	}
	offsets, total := make([]int, len(buffers)), 0
	for index, buffer := range buffers {
		offsets[index] = total
		total += (buffer.count + frozenSlabAlignment - 1) / frozenSlabAlignment * frozenSlabAlignment
	}
	// Every count and offset then fits the kernels' 32-bit words and cuBLAS's dimensions.
	if total > math.MaxInt32 {
		return fmt.Errorf("%s: %d values exceed one 32-bit-indexed allocation", name, total)
	}
	slab, err := AllocResidentF32(s.worker, total, nil)
	if err != nil {
		return err
	}
	l.slab = slab
	for index, buffer := range buffers {
		*buffer.pointer = ptrAt(slab, offsets[index])
	}
	err = s.worker.Do(context.Background(), func(state *device.State) error {
		for _, weight := range weights {
			if err := state.Driver.MemcpyHtoD(*weight.pointer, driver.Bytes(weight.data)); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return errors.Join(err, l.Close())
	}
	return nil
}

// Forward runs the layer over x, Tokens rows of Hidden values on the card, and returns its output, a buffer of the
// layer's that holds until the next Forward. It keeps on the card what Backward reads.
func (l *frozenLayer) Forward(x driver.DevicePtr) (driver.DevicePtr, error) {
	if l.closed || x == 0 {
		return 0, fmt.Errorf("%s: closed, or no input", l.name)
	}
	l.forwarded = false
	if err := l.run(l.forward, x); err != nil {
		return 0, err
	}
	l.forwarded = true
	return l.around.out, nil
}

// Backward takes dOut, the gradient with respect to the last Forward's output on the card, and returns the gradient
// with respect to that forward's input, a buffer of the layer's that holds until the next Backward. No weight's
// gradient is formed.
func (l *frozenLayer) Backward(dOut driver.DevicePtr) (driver.DevicePtr, error) {
	if l.closed || !l.forwarded || dOut == 0 {
		return 0, fmt.Errorf("%s: closed, no forward to walk back, or no output gradient", l.name)
	}
	if err := l.run(l.backward, dOut); err != nil {
		return 0, err
	}
	return l.around.dx, nil
}

// run queues pass over input in one resident scope of the session and waits for it.
func (l *frozenLayer) run(pass func(*frozenRun, driver.DevicePtr), input driver.DevicePtr) error {
	return l.session.Run(func(ops *ResidentOps) error {
		r := frozenRun{ops: ops}
		pass(&r, input)
		return r.err
	})
}

// Close releases the layer's buffers; the session stays open.
func (l *frozenLayer) Close() error {
	if l.closed {
		return nil
	}
	l.closed = true
	return FreeResident(l.session.worker, l.slab)
}

// forward queues hostmath.HybridDecoderLayerForward around the layer's mix.
func (l *frozenLayer) forward(r *frozenRun, x driver.DevicePtr) {
	d, a := l.d, &l.around
	tokens, hidden, inter := d.Tokens, d.Hidden, d.Inter
	r.each("copy_f32", tokens*hidden, x, a.x, tokens*hidden)
	r.rows("weighted_rms_norm_f32", tokens, a.x, a.inputNorm, a.normed, hidden, tokens, 1, d.Eps)
	l.mix.forwardMix(r)
	r.each("add_f32", tokens*hidden, a.x, a.mixed, a.h, tokens*hidden)
	r.rows("weighted_rms_norm_f32", tokens, a.h, a.postNorm, a.normed, hidden, tokens, 1, d.Eps)
	r.project(a.normed, a.gate, a.mlpGate, tokens, hidden, inter)
	r.project(a.normed, a.up, a.mlpUp, tokens, hidden, inter)
	r.each("silu_f32", tokens*inter, a.mlpGate, a.inter, tokens*inter)
	r.each("multiply_f32", tokens*inter, a.inter, a.mlpUp, a.inter, tokens*inter)
	r.project(a.inter, a.down, a.mixed, tokens, inter, hidden)
	r.each("add_f32", tokens*hidden, a.h, a.mixed, a.out, tokens*hidden)
}

// backward queues the VJP of forward with respect to its input alone, as hostmath.HybridDecoderLayerBackward gives
// DX, from the cache the last forward kept.
func (l *frozenLayer) backward(r *frozenRun, dOut driver.DevicePtr) {
	d, a := l.d, &l.around
	tokens, hidden, inter := d.Tokens, d.Hidden, d.Inter
	// out = h + Down·(SiLU(gate)⊙up): dm = dOut·Down, then the gate's and up's gradients, then back to hn.
	r.back(dOut, a.down, a.inter2, tokens, hidden, inter, false)
	r.each("silu_f32", tokens*inter, a.mlpGate, a.inter, tokens*inter)
	r.each("multiply_f32", tokens*inter, a.inter2, a.inter, a.inter, tokens*inter)
	r.each("multiply_f32", tokens*inter, a.inter2, a.mlpUp, a.inter2, tokens*inter)
	r.each("silu_backward_f32", tokens*inter, a.inter2, a.mlpGate, a.inter2, tokens*inter)
	r.back(a.inter2, a.gate, a.normed, tokens, inter, hidden, false)
	r.back(a.inter, a.up, a.normed, tokens, inter, hidden, true)
	// dh = dOut + the post-norm's input gradient.
	r.rows("rms_norm_input_backward_f32", tokens, a.normed, a.h, a.postNorm, a.mixed, hidden, tokens, d.Eps)
	r.each("add_f32", tokens*hidden, a.mixed, dOut, a.mixed, tokens*hidden)
	l.mix.backwardMix(r)
	// dx = dh + the input norm's input gradient.
	r.rows("rms_norm_input_backward_f32", tokens, a.normed, a.x, a.inputNorm, a.dx, hidden, tokens, d.Eps)
	r.each("add_f32", tokens*hidden, a.dx, a.mixed, a.dx, tokens*hidden)
}

// frozenRun queues one pass's device work in a resident scope and keeps its first failure, after which it queues
// nothing more.
type frozenRun struct {
	ops *ResidentOps
	err error
}

// kernel launches the module's named kernel over grid blocks of the default threads. Each value is a device pointer,
// an int passed as a 32-bit word, or a float64 passed as a float32.
func (r *frozenRun) kernel(name string, grid driver.Dim3, values ...any) {
	if r.err != nil {
		return
	}
	function, err := r.ops.function(name)
	if err != nil {
		r.err = err
		return
	}
	arguments := make([]unsafe.Pointer, len(values))
	for index, value := range values {
		switch typed := value.(type) {
		case driver.DevicePtr:
			arguments[index] = unsafe.Pointer(&typed)
		case int:
			word := uint32(typed)
			arguments[index] = unsafe.Pointer(&word)
		case float64:
			word := float32(typed)
			arguments[index] = unsafe.Pointer(&word)
		default:
			r.err = fmt.Errorf("frozen layer: %s takes no %T", name, value)
			return
		}
	}
	state := r.ops.session.state
	r.err = state.Driver.LaunchKernel(function, grid, driver.Dim3{X: deviceBlockThreads, Y: 1, Z: 1}, 0, state.Stream, arguments)
}

// each launches name with a thread for each of count items.
func (r *frozenRun) each(name string, count int, values ...any) {
	blocks := (uint32(count) + deviceBlockThreads - 1) / deviceBlockThreads
	r.kernel(name, driver.Dim3{X: blocks, Y: 1, Z: 1}, values...)
}

// rows launches name with a block for each of count rows.
func (r *frozenRun) rows(name string, count int, values ...any) {
	r.kernel(name, driver.Dim3{X: uint32(count), Y: 1, Z: 1}, values...)
}

// zero clears count values of buffer.
func (r *frozenRun) zero(buffer driver.DevicePtr, count int) {
	if r.err == nil {
		r.err = r.ops.Zero(buffer, count)
	}
}

// project writes output = input·weightᵀ, input [rows, in] and weight [out, in], as hostmath.Linear does.
func (r *frozenRun) project(input, weight, output driver.DevicePtr, rows, in, out int) {
	if r.err == nil {
		r.err = r.ops.GEMM(false, true, rows, in, out, input, weight, output)
	}
}

// back writes project's input gradient, output = gradient·weight for gradient [rows, out], or adds it to output when
// accumulate is set.
func (r *frozenRun) back(gradient, weight, output driver.DevicePtr, rows, out, in int, accumulate bool) {
	if r.err != nil {
		return
	}
	kept := float32(0)
	if accumulate {
		kept = 1
	}
	// cuBLAS reads the row-major operands as their column-major transposes: outputᵀ = weightᵀ·gradientᵀ.
	blas := r.ops.session
	r.err = blas.library.GEMMEx(blas.handle, cublas.OperationNone, cublas.OperationNone, int32(in), int32(rows), int32(out), 1,
		weight, cublas.DataF32, int32(in), gradient, cublas.DataF32, int32(out), kept, output, cublas.DataF32, int32(in),
		cublas.ComputeF32, cublas.GemmDefault)
}

// batched writes batch products output[b] = left[b]·right[b] of rows, inner and columns, each operand transposed as
// asked and its matrices strides apart; a stride of 0 reads one matrix for every product.
func (r *frozenRun) batched(transposeLeft, transposeRight bool, rows, inner, columns, batch int, left driver.DevicePtr, leftStride int, right driver.DevicePtr, rightStride int, output driver.DevicePtr, outputStride int) {
	if r.err == nil {
		r.err = r.ops.session.gemmStrided(transposeLeft, transposeRight, rows, inner, columns, batch, left, leftStride, right, rightStride, output, outputStride)
	}
}
