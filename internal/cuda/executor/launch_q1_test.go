package executor

import (
	"encoding/binary"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestQ1MulMatLaunchesOneWarpPerOutputOverWholeBlocks probes the launch
// mul_mat_q1_0_f32 is written against: one warp per weight row and input
// column in 256-thread blocks, the width passed as it is, and a width that is
// not whole 128-column blocks refused before any launch. The kernel reads a
// row block by block and relies on both.
func TestQ1MulMatLaunchesOneWarpPerOutputOverWholeBlocks(t *testing.T) {
	const output, left, right = 1 << 40, 2 << 40, 3 << 40
	var functions functionSet
	functions[kernelMulMatQ10F32] = boundKernel{
		id: kernelMulMatQ10F32, argumentCount: kernelFunctionArgumentCounts[kernelMulMatQ10F32],
	}
	probe := func(inner, rows, columns uint64) ([]byte, error) {
		builder := tensor.NewBuilder()
		weight := builder.Input("weight", dtype.Q1_0, tensor.MustShape(inner, rows))
		input := builder.Input("input", dtype.F32, tensor.MustShape(inner, columns))
		node := builder.MulMat(weight, input)
		if err := builder.Err(); err != nil {
			return nil, err
		}
		// Slot 0 is the output; slots 1 and 2 are the weight and the input.
		frame := launchPointerFrame{values: []driver.DevicePtr{output, left, right}, slots: []int{0, 1, 2}}
		trace := &device.LaunchTrace{Probe: true}
		err := launchLinearLayout(&device.State{Trace: trace}, functions, nil, nil, node, node.Attrs, frame)
		return trace.Data(), err
	}
	const inner, rows, columns = 384, 13, 5
	raw, err := probe(inner, rows, columns)
	if err != nil {
		t.Fatal(err)
	}
	// A record is four header words, then each argument: eight bytes a
	// pointer, four a scalar.
	const recordBytes = 4*8 + 3*8 + 3*4
	if len(raw) != recordBytes {
		t.Fatalf("launch bytes %d, want one launch (%d)", len(raw), recordBytes)
	}
	word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
	const warpThreads, blockThreads = 32, 256
	for _, check := range []struct {
		name      string
		got, want uint64
	}{
		{"kernel", word(raw), uint64(kernelMulMatQ10F32)},
		{"grid", word(raw[8:]), (rows*columns*warpThreads+blockThreads-1)/blockThreads | 1<<32 | 1<<48},
		{"block", word(raw[16:]), blockThreads | 1<<32 | 1<<48},
		{"shared bytes", word(raw[24:]), 0},
		{"weight", word(raw[32:]), left},
		{"input", word(raw[40:]), right},
		{"output", word(raw[48:]), output},
		{"width", uint64(scalar(raw[56:])), inner},
		{"rows", uint64(scalar(raw[60:])), rows},
		{"columns", uint64(scalar(raw[64:])), columns},
	} {
		if check.got != check.want {
			t.Fatalf("%s = %#x, want %#x", check.name, check.got, check.want)
		}
	}
	if raw, err := probe(100, rows, 1); err == nil || len(raw) != 0 {
		t.Fatalf("a width of 100 columns launched %d bytes (error %v), want a refusal", len(raw), err)
	}
}
