//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// float64Words is how many float32 words of a scope buffer hold one float64 scratch value.
//   - source = the sizes of float64 and float32
//   - scope = the float64 scratch of the gated delta-net and short-convolution backward kernels
//   - reconsider when the scope gains a buffer typed by its element
const float64Words = int(unsafe.Sizeof(float64(0)) / unsafe.Sizeof(float32(0)))

// GatedDeltaNetBackwardDevice is the cgo-free CUDA port of
// hostmath.GatedDeltaNetBackward: the BPTT VJP of the gated delta-net recurrence.
// It launches gated_delta_net_backward_f32 with one block per (sequence,head) and
// one thread per state row (the token loop is sequential per row; rows are
// independent). Internal math runs in double to mirror the host f64 reference;
// grouped q/k, dgate, and dbeta accumulate via atomicAdd into zero-initialized
// buffers. Two f64 scratch buffers carry the recomputed forward trajectory and
// the reverse carry. Returns the same six gradients as the host, in the same
// order.
func GatedDeltaNetBackwardDevice(
	worker *device.Worker,
	query, key, value, gate, beta, inputState, dOutput []float32,
	size, queryHeads, keyHeads, heads, tokens, sequences, gateWidth int,
	repeatInterleave bool,
) (dQuery, dKey, dValue, dGate, dBeta, dInputState []float32, err error) {
	if size <= 0 || queryHeads <= 0 || keyHeads <= 0 || heads <= 0 ||
		tokens <= 0 || sequences <= 0 || gateWidth <= 0 {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf(
			"GatedDeltaNetBackwardDevice: nonpositive dimension")
	}
	attn := size * heads * tokens * sequences
	wantQuery := size * queryHeads * tokens * sequences
	wantKey := size * keyHeads * tokens * sequences
	wantGate := gateWidth * heads * tokens * sequences
	wantBeta := heads * tokens * sequences
	wantState := heads * sequences * size * size
	if len(query) != wantQuery || len(key) != wantKey || len(value) != attn ||
		len(gate) != wantGate || len(beta) != wantBeta ||
		len(inputState) != wantState || len(dOutput) != attn {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf(
			"GatedDeltaNetBackwardDevice: shape mismatch")
	}

	dQuery = make([]float32, wantQuery)
	dKey = make([]float32, wantKey)
	dValue = make([]float32, attn)
	dGate = make([]float32, wantGate)
	dBeta = make([]float32, wantBeta)
	dInputState = make([]float32, wantState)

	count := heads * sequences

	err = withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("gated_delta_net_backward_f32")
		if err != nil {
			return err
		}
		// Outputs zero-initialized: uploading zeroed Go slices both allocs and clears.
		var qP, kP, vP, gP, bP, sP, doP, dqP, dkP, dvP, dgP, dbP, disP, safterP, dsCarryP driver.DevicePtr
		if err := scope.fill([]deviceSlot{
			{&qP, query, 0}, {&kP, key, 0}, {&vP, value, 0}, {&gP, gate, 0}, {&bP, beta, 0}, {&sP, inputState, 0}, {&doP, dOutput, 0},
			{&dqP, dQuery, 0}, {&dkP, dKey, 0}, {&dvP, dValue, 0}, {&dgP, dGate, 0}, {&dbP, dBeta, 0}, {&disP, dInputState, 0},
			{&safterP, nil, count * size * tokens * size * float64Words}, {&dsCarryP, nil, count * size * size * float64Words},
		}); err != nil {
			return err
		}

		sz, qh, kh, hd := uint32(size), uint32(queryHeads), uint32(keyHeads), uint32(heads)
		tk, sq, gw := uint32(tokens), uint32(sequences), uint32(gateWidth)
		ri := uint32(0)
		if repeatInterleave {
			ri = 1
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&gP),
			unsafe.Pointer(&bP), unsafe.Pointer(&sP), unsafe.Pointer(&doP),
			unsafe.Pointer(&dqP), unsafe.Pointer(&dkP), unsafe.Pointer(&dvP),
			unsafe.Pointer(&dgP), unsafe.Pointer(&dbP), unsafe.Pointer(&disP),
			unsafe.Pointer(&safterP), unsafe.Pointer(&dsCarryP),
			unsafe.Pointer(&sz), unsafe.Pointer(&qh), unsafe.Pointer(&kh), unsafe.Pointer(&hd),
			unsafe.Pointer(&tk), unsafe.Pointer(&sq), unsafe.Pointer(&gw), unsafe.Pointer(&ri),
		}
		if err := scope.state.Driver.LaunchKernel(fn,
			driver.Dim3{X: uint32(count), Y: 1, Z: 1},
			driver.Dim3{X: deviceBlockThreads, Y: 1, Z: 1},
			0, scope.state.Stream, args); err != nil {
			return err
		}
		return scope.finish(
			cudaDownload{dQuery, dqP}, cudaDownload{dKey, dkP}, cudaDownload{dValue, dvP},
			cudaDownload{dGate, dgP}, cudaDownload{dBeta, dbP}, cudaDownload{dInputState, disP},
		)
	})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	return dQuery, dKey, dValue, dGate, dBeta, dInputState, nil
}
