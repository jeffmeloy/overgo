//go:build windows

package executor

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestCellAttentionMatchesReference holds the cell attention operation (tensor.CellAttention) on the device to the
// reference backend's over the same inputs, step after step on one state behind a shuffled page table: a prompt chunk
// past a sliding window (the int8 prompt attention), single tokens and a verify window of three (the decode chunks),
// each step's attention within 1e-3 of the output scale of the reference's float64 attention over the dequantized cells
// and the state's codes and fp16 scales equal to the reference's bit for bit at the end. The geometries are gemma-4-12B's
// sliding layers (8 K/V heads of 256 reading a window, here 192 cells), its global layers (one K/V head of 512 reading
// every cell before it) and a llama layer (2 K/V heads of 128).
func TestCellAttentionMatchesReference(t *testing.T) {
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	for _, c := range []struct {
		name                          string
		width, heads, kvHeads, window uint64
	}{
		{"gemma-4-12B sliding", 256, 16, 8, 192},
		{"gemma-4-12B global", 512, 16, 1, 0},
		{"llama", 128, 16, 2, 0},
	} {
		t.Run(c.name, func(t *testing.T) { cellAttentionParity(t, cuda, c.width, c.heads, c.kvHeads, c.window) })
	}
}

func cellAttentionParity(t *testing.T, cuda *Executor, width, heads, kvHeads, window uint64) {
	const capacity, prompt = 1024, 400
	ctx := context.WithoutCancel(t.Context())
	draws := rand.New(rand.NewPCG(width, heads))
	pages := uint64(capacity / tensor.CellPageCells)
	cells := kvHeads * capacity
	codeShape, scaleShape := tensor.MustShape(width, cells), tensor.MustShape(width/tensor.QSAScaleGroup, cells)
	table := make([]float32, pages)
	tableWords := make([]int32, pages)
	for index, page := range draws.Perm(int(pages)) {
		table[index], tableWords[index] = float32(page), int32(page)
	}
	kinds := []struct {
		kind  dtype.Type
		shape tensor.Shape
	}{{dtype.I8, codeShape}, {dtype.I8, codeShape}, {dtype.F16, scaleShape}, {dtype.F16, scaleShape}, {dtype.I32, tensor.MustShape(pages)}}
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	hostState := make([]reference.Value, len(kinds))
	devicePointers := make([]driver.DevicePtr, len(kinds))
	for index, part := range kinds {
		elements, err := part.shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		bytes, err := part.shape.Bytes(part.kind)
		if err != nil {
			t.Fatal(err)
		}
		hostState[index] = reference.Value{Shape: part.shape, Data: make([]float32, elements)}
		devicePointers[index] = buffers.zeros(t, int(bytes))
	}
	copy(hostState[4].Data, table)
	buffers.write(t, devicePointers[4], int32Bytes(tableWords))
	scale := float32(1 / math.Sqrt(float64(width)))
	step := func(first, tokens uint64) {
		t.Helper()
		cellsOf := make([]uint32, tokens)
		for index := range cellsOf {
			cellsOf[index] = uint32(first + uint64(index))
		}
		attributes := tensor.CellAttentionAttributes{Cells: cellsOf, Capacity: capacity, Window: uint32(window), Scale: scale}
		random := func(count uint64) []float32 {
			values := make([]float32, count)
			for index := range values {
				values[index] = float32(draws.NormFloat64())
			}
			return values
		}
		inputs := []reference.Value{
			{Shape: tensor.MustShape(width, heads, tokens), Data: random(width * heads * tokens)},
			{Shape: tensor.MustShape(width, kvHeads, tokens), Data: random(width * kvHeads * tokens)},
			{Shape: tensor.MustShape(width, kvHeads, tokens), Data: random(width * kvHeads * tokens)},
		}
		build := func() (*tensor.Tensor, []*tensor.Tensor, []*tensor.Tensor) {
			builder := tensor.NewBuilder()
			var rows []*tensor.Tensor
			for index, name := range []string{"query", "key", "value"} {
				rows = append(rows, builder.Input(name, dtype.F32, inputs[index].Shape))
			}
			state := make([]*tensor.Tensor, len(kinds))
			for index, part := range kinds {
				state[index] = builder.Input([]string{"key_codes", "value_codes", "key_scales", "value_scales", "page_table"}[index], part.kind, part.shape)
			}
			output := builder.CellAttention(rows[0], rows[1], rows[2], tensor.CellState{
				KeyCodes: state[0], ValueCodes: state[1], KeyScales: state[2], ValueScales: state[3], PageTable: state[4],
			}, attributes)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			return output, rows, state
		}
		output, rows, state := build()
		host := map[*tensor.Tensor]reference.Value{}
		for index, node := range rows {
			host[node] = inputs[index]
		}
		device := map[*tensor.Tensor]driver.DevicePtr{}
		for index, node := range state {
			device[node] = devicePointers[index]
		}
		got, err := cuda.executeWithDeviceFeeds(ctx, []*tensor.Tensor{output}, host, device)
		if err != nil {
			t.Fatal(err)
		}
		referenceOutput, referenceRows, referenceState := build()
		feeds := map[*tensor.Tensor]reference.Value{}
		for index, node := range referenceRows {
			feeds[node] = inputs[index]
		}
		for index, node := range referenceState {
			feeds[node] = hostState[index]
		}
		want, err := reference.Execute([]*tensor.Tensor{referenceOutput}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		difference, outputScale := 0.0, 0.0
		for index, value := range want[referenceOutput].Data {
			difference = max(difference, math.Abs(float64(got[output].Data[index]-value)))
			outputScale = max(outputScale, math.Abs(float64(value)))
		}
		if !(difference <= 1e-3*outputScale) {
			t.Fatalf("%d tokens from cell %d: the device stands %.3g from the reference, past 1e-3 of the output scale %.3g",
				tokens, first, difference, outputScale)
		}
	}
	step(0, prompt)
	for cell := uint64(prompt); cell < prompt+4; cell++ {
		step(cell, 1)
	}
	step(prompt+4, 3)
	for index, name := range []string{"key codes", "value codes"} {
		codes := buffers.read(t, devicePointers[index], int(width*cells))
		for at, code := range codes {
			if float32(int8(code)) != hostState[index].Data[at] {
				t.Fatalf("%s differ at %d: device %d, reference %v", name, at, int8(code), hostState[index].Data[at])
			}
		}
	}
	for index, name := range []string{"key scales", "value scales"} {
		slot := index + 2
		scales := uint16s(buffers.read(t, devicePointers[slot], int(2*width/tensor.QSAScaleGroup*cells)))
		for at, bits := range scales {
			if bits != quant.Float32ToFloat16(hostState[slot].Data[at]) {
				t.Fatalf("%s differ at %d: device %#x, reference %v", name, at, bits, hostState[slot].Data[at])
			}
		}
	}
}
