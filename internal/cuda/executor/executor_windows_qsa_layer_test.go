//go:build windows

package executor

import (
	"context"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestQSALayerMatchesStrata holds the QSA attention operation - Strata's QSA layer between its projections, its
// ported kernels in Strata's order - to the reference backend's QSA unit over the same inputs: a prompt chunk past the
// selection budget (Strata's prompt read: the chunk's cells at once, the native batch indexer, the tensor-core
// scores and attention), then decode steps (its decode layer: the table indexer, the multi-query scores, the split-K
// attention), each step's attention within a bound of the reference's, and the state's int8 codes and fp16 scales
// equal to the reference's bit for bit at the end. The state sits behind a reversed page table.
func TestQSALayerMatchesStrata(t *testing.T) {
	const (
		capacity = uint32(8192)
		prompt   = 3000
		decode   = 6
		// A picture of 10 x 12 patches from cell 120, the way Strata's picture request lays its cells out (generate.cpp
		// 10736-10800): its cells at (p, p + y, p + x), the text after it from p + 12.
		picture, pictureRows, pictureColumns = 120, 10, 12
		// The selection this layer test holds both sides to. Each block's indexer key stands on a rung of its own, an
		// integer FP16 keeps exactly, in a dimension no rotation touches, beside an anchor that sets its norm; each
		// query looks along that dimension far harder than along its random rest. Block scores then stand a rung
		// apart, past 0.08 percent of a score against FP32's rounding of it, so every boundary is separated and both
		// sides select the same cells. The stride scatters the selected blocks. A near tie is strataBlockTopKParity's
		// to judge, and attention over given cells strataAttentionParity's.
		rungs, rungStride              = (prompt+decode)/tensor.QSABlockCells + 1, 311
		rungDimension, anchorDimension = tensor.QSARotated, tensor.QSARotated + 1
		rungAnchor, rungQuery          = 1024, 100
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	ctx := context.WithoutCancel(t.Context())
	draws := rand.New(rand.NewPCG(41, 43))
	pages := uint64(capacity) / tensor.QSABlockCells

	// The state, on the device and as the reference's values.
	shapes := qsaStateShapes(uint64(capacity))
	table := make([]float32, pages)
	tableWords := make([]int32, pages)
	for page := range pages {
		table[page], tableWords[page] = float32(pages-1-page), int32(pages-1-page)
	}
	cos, sin := tensor.QSARopeTable(int(capacity))
	hostState := make([]reference.Value, len(shapes))
	devicePointers := make([]driver.DevicePtr, len(shapes))
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	for index, shape := range shapes {
		elements, err := shape.shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		hostState[index] = reference.Value{Shape: shape.shape, Data: make([]float32, elements)}
		bytes, err := shape.shape.Bytes(shape.kind)
		if err != nil {
			t.Fatal(err)
		}
		devicePointers[index] = buffers.zeros(t, int(bytes))
	}
	copy(hostState[qsaInputPageTable-qsaInputKeyCodes].Data, table)
	buffers.write(t, devicePointers[qsaInputPageTable-qsaInputKeyCodes], int32Bytes(tableWords))
	copy(hostState[qsaInputRopeCos-qsaInputKeyCodes].Data, cos)
	copy(hostState[qsaInputRopeSin-qsaInputKeyCodes].Data, sin)
	buffers.write(t, devicePointers[qsaInputRopeCos-qsaInputKeyCodes], float32Bytes(cos))
	buffers.write(t, devicePointers[qsaInputRopeSin-qsaInputKeyCodes], float32Bytes(sin))
	positionWords := make([]int32, tensor.QSAPositionAxes*capacity)
	for cell := range int32(capacity) {
		row := [tensor.QSAPositionAxes]int32{cell, cell, cell}
		switch picture := int32(picture); {
		case cell >= picture+pictureRows*pictureColumns:
			p := cell - pictureRows*pictureColumns + pictureColumns
			row = [tensor.QSAPositionAxes]int32{p, p, p}
		case cell >= picture:
			patch := cell - picture
			row = [tensor.QSAPositionAxes]int32{picture, picture + patch/pictureColumns, picture + patch%pictureColumns}
		}
		for axis, position := range row {
			positionWords[int(cell)*tensor.QSAPositionAxes+axis] = position
			hostState[qsaInputPositions-qsaInputKeyCodes].Data[int(cell)*tensor.QSAPositionAxes+axis] = float32(position)
		}
	}
	buffers.write(t, devicePointers[qsaInputPositions-qsaInputKeyCodes], int32Bytes(positionWords))
	// The fixed state each side carries from step to step, as a cache holds it: the tail and the spare after the last.
	deviceTail, deviceSpare := make([]float32, tensor.QSATailRows*tensor.QSAIndexerWidth), make([]float32, tensor.QSAIndexerWidth)
	referenceTail, referenceSpare := slices.Clone(deviceTail), slices.Clone(deviceSpare)
	norm := make([]float32, tensor.QSAIndexerWidth)
	for index := range norm {
		norm[index] = 1 + float32(0.1*draws.NormFloat64())
	}

	random := func(count uint64, scale float64) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64() * scale)
		}
		return values
	}
	step := func(first uint32, tokens uint32) {
		t.Helper()
		cells := make([]uint32, tokens)
		for index := range cells {
			cells[index] = first + uint32(index)
		}
		n := uint64(tokens)
		indexerKey := random(tensor.QSAIndexerWidth*n, 1)
		indexerQuery := random(tensor.QSAIndexerWidth*tensor.QSAIndexerHeads*n, 0.2)
		for token, cell := range cells {
			key := indexerKey[token*tensor.QSAIndexerWidth:]
			key[rungDimension], key[anchorDimension] = float32(cell/tensor.QSABlockCells*rungStride%rungs), rungAnchor
			for head := range tensor.QSAIndexerHeads {
				indexerQuery[(token*tensor.QSAIndexerHeads+head)*tensor.QSAIndexerWidth+rungDimension] = rungQuery
			}
		}
		inputs := map[string]reference.Value{
			"query":         {Shape: tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAQueryHeads, n), Data: random(tensor.QSAHeadWidth*tensor.QSAQueryHeads*n, 1)},
			"key":           {Shape: tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, n), Data: random(tensor.QSAHeadWidth*tensor.QSAKeyValueHeads*n, 1)},
			"value":         {Shape: tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, n), Data: random(tensor.QSAHeadWidth*tensor.QSAKeyValueHeads*n, 1)},
			"indexer_key":   {Shape: tensor.MustShape(tensor.QSAIndexerWidth, 1, n), Data: indexerKey},
			"indexer_query": {Shape: tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSAIndexerHeads, n), Data: indexerQuery},
			"indexer_norm":  {Shape: tensor.MustShape(tensor.QSAIndexerWidth), Data: norm},
		}
		attributes := tensor.QSAAttentionAttributes{Cells: cells, Capacity: capacity, Epsilon: strataRMSEpsilon}
		builder := tensor.NewBuilder()
		named := map[string]*tensor.Tensor{}
		for _, name := range []string{"query", "key", "value", "indexer_key", "indexer_query", "indexer_norm"} {
			named[name] = builder.Input(name, dtype.F32, inputs[name].Shape)
		}
		tail := builder.Input("indexer_tail", dtype.F32, tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSATailRows))
		spare := builder.Input("indexer_spare", dtype.F32, tensor.MustShape(tensor.QSAIndexerWidth))
		named["indexer_tail"], named["indexer_spare"] = tail, spare
		state := make([]*tensor.Tensor, len(shapes))
		for index, shape := range shapes {
			state[index] = builder.Input(shape.name, shape.kind, shape.shape)
		}
		output := builder.QSAAttention(named["query"], named["key"], named["value"], named["indexer_key"], named["indexer_query"], named["indexer_norm"],
			tail, spare, tensor.QSAState{
				KeyCodes: state[0], ValueCodes: state[1], KeyScales: state[2], ValueScales: state[3], PageTable: state[4],
				IndexerPooled: state[5], BlockPosition: state[6], Positions: state[7], RopeCos: state[8], RopeSin: state[9],
			}, attributes)
		testcheck.NoError(t, builder.Err())
		inputs["indexer_tail"], inputs["indexer_spare"] = reference.Value{Shape: tail.Shape, Data: deviceTail}, reference.Value{Shape: spare.Shape, Data: deviceSpare}
		host := map[*tensor.Tensor]reference.Value{}
		for name, node := range named {
			host[node] = inputs[name]
		}
		device := map[*tensor.Tensor]driver.DevicePtr{}
		for index, node := range state {
			device[node] = devicePointers[index]
		}
		got := testcheck.Must(cuda.executeWithDeviceFeeds(ctx, []*tensor.Tensor{output}, host, device))(t)
		feeds := maps.Clone(host)
		feeds[tail], feeds[spare] = reference.Value{Shape: tail.Shape, Data: referenceTail}, reference.Value{Shape: spare.Shape, Data: referenceSpare}
		for index, node := range state {
			feeds[node] = hostState[index]
		}
		want := testcheck.Must(reference.Execute([]*tensor.Tensor{output}, feeds))(t)
		// The packed output: the attention, then the fixed state each side carries to its next step.
		attentionValues := tensor.QSAHeadWidth * tensor.QSAQueryHeads * int(tokens)
		gotFixed, wantFixed := got[output].Data[attentionValues:], want[output].Data[attentionValues:]
		copy(deviceTail, gotFixed)
		copy(deviceSpare, gotFixed[len(deviceTail):])
		copy(referenceTail, wantFixed)
		copy(referenceSpare, wantFixed[len(referenceTail):])
		difference, _ := maxAbsDifference(t, got[output].Data[:attentionValues], want[output].Data[:attentionValues])
		scale := 0.0
		for _, value := range want[output].Data[:attentionValues] {
			scale = max(scale, math.Abs(float64(value)))
		}
		// Both sides selected the same cells (rungs above), so the attention differs from the reference's by the kernels'
		// rounding alone (FP32 for a token, FP16 hi/lo products with FP32 sums for a chunk, against float64): the prompt
		// parity's 1e-4 of the output scale.
		if !(difference <= 1e-4*scale) {
			t.Fatalf("cells %d..%d: the attention stands %.3g from the reference, past 1e-4 of the scale %.3g", first, first+tokens-1, difference, scale)
		}
		// The rotated half of each pooled key is all the picture's positions reach, and the rungs do not read it. Each side
		// takes a float32 angle of position * theta^pair, and this fixture's positions pass 2048, where a float32 angle
		// rounds to 2.4e-4 of a radian: a rotated value may stand two such shares of the row's size from the reference's.
		devicePooled := float32s(buffers.read(t, devicePointers[qsaInputIndexerPooled-qsaInputKeyCodes], int(4*tensor.QSAIndexerWidth*(pages+tensor.QSAPooledSpare))))
		referencePooled := hostState[qsaInputIndexerPooled-qsaInputKeyCodes].Data
		pooledDifference, pooledScale := 0.0, 0.0
		for block := range uint64(first+tokens) / tensor.QSABlockCells {
			rotated := referencePooled[block*tensor.QSAIndexerWidth:][:tensor.QSARotated]
			apart, _ := maxAbsDifference(t, devicePooled[block*tensor.QSAIndexerWidth:][:tensor.QSARotated], rotated)
			pooledDifference = max(pooledDifference, apart)
			for _, value := range rotated {
				pooledScale = max(pooledScale, math.Abs(float64(value)))
			}
		}
		if !(pooledDifference <= 5e-4*pooledScale) {
			t.Fatalf("cells %d..%d: the rotated pooled keys stand %.3g from the reference, past 5e-4 of the scale %.3g", first, first+tokens-1, pooledDifference, pooledScale)
		}
		t.Logf("cells %d..%d: %.3g from the reference (scale %.3g); rotated pooled keys %.3g apart (scale %.3g)",
			first, first+tokens-1, difference, scale, pooledDifference, pooledScale)
	}
	step(0, prompt)
	for token := range uint32(decode) {
		step(prompt+token, 1)
	}

	// The tail each side carries: the raw keys of the incomplete block's cells, bit for bit.
	for at, value := range deviceTail {
		if math.Float32bits(value) != math.Float32bits(referenceTail[at]) {
			t.Fatalf("the indexer tail differs at %d: device %v, reference %v", at, value, referenceTail[at])
		}
	}
	// The state: the int8 codes and the fp16 scales bit for bit.
	held := prompt + decode
	for index, name := range []string{"key codes", "value codes"} {
		codes := buffers.read(t, devicePointers[index], int(tensor.QSAHeadWidth*tensor.QSAKeyValueHeads*uint64(capacity)))
		for cell := range held {
			page := tableWords[cell/tensor.QSABlockCells]
			for head := range tensor.QSAKeyValueHeads {
				row := (int(page)*tensor.QSAKeyValueHeads+head)*tensor.QSABlockCells + cell%tensor.QSABlockCells
				for d := range tensor.QSAHeadWidth {
					at := row*tensor.QSAHeadWidth + d
					if int8(codes[at]) != int8(hostState[index].Data[at]) {
						t.Fatalf("%s of cell %d head %d differ at %d: device %d, reference %v", name, cell, head, d, int8(codes[at]), hostState[index].Data[at])
					}
				}
			}
		}
	}
	for index, name := range []string{"key scales", "value scales"} {
		slot := index + 2
		scales := uint16s(buffers.read(t, devicePointers[slot], int(2*tensor.QSAHeadWidth/tensor.QSAScaleGroup*tensor.QSAKeyValueHeads*uint64(capacity))))
		for at, bits := range scales {
			if bits != quant.Float32ToFloat16(hostState[slot].Data[at]) {
				t.Fatalf("%s differ at %d: device %#x, reference %v", name, at, bits, hostState[slot].Data[at])
			}
		}
	}
}

// TestQSAWindowStepIsCaptured holds a verify window's QSA step to a captured graph, as verify.cpp record_window
// captures each window: its rows read their staged step records, so its next cells reach no kernel as launch
// arguments, while a step of several tokens over a copy of the fixed state passes its first cell and active blocks as
// launch arguments, which keeps its graph from being captured.
func TestQSAWindowStepIsCaptured(t *testing.T) {
	const capacity = 1024
	for _, inPlace := range []bool{false, true} {
		builder := tensor.NewBuilder()
		input := func(name string, shape tensor.Shape) *tensor.Tensor { return builder.Input(name, dtype.F32, shape) }
		query := input("query", tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAQueryHeads, 2))
		key := input("key", tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, 2))
		value := input("value", tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads, 2))
		indexerKey := input("indexer_key", tensor.MustShape(tensor.QSAIndexerWidth, 1, 2))
		indexerQuery := input("indexer_query", tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSAIndexerHeads, 2))
		indexerNorm := input("indexer_norm", tensor.MustShape(tensor.QSAIndexerWidth))
		tail := input("indexer_tail", tensor.MustShape(tensor.QSAIndexerWidth, tensor.QSATailRows))
		spare := input("indexer_spare", tensor.MustShape(tensor.QSAIndexerWidth))
		var state []*tensor.Tensor
		for _, shape := range qsaStateShapes(capacity) {
			state = append(state, builder.Input(shape.name, shape.kind, shape.shape))
		}
		qsaState := tensor.QSAState{
			KeyCodes: state[0], ValueCodes: state[1], KeyScales: state[2], ValueScales: state[3], PageTable: state[4],
			IndexerPooled: state[5], BlockPosition: state[6], Positions: state[7], RopeCos: state[8], RopeSin: state[9],
		}
		attributes := tensor.QSAAttentionAttributes{Cells: []uint32{5, 6}, Capacity: capacity, Epsilon: strataRMSEpsilon}
		output := builder.QSAAttention(query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare, qsaState, attributes)
		if inPlace {
			snapshot := input("indexer_snapshot", tail.Shape)
			output = builder.QSAAttentionInPlace(query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare, snapshot,
				qsaState, attributes)
		}
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		compiled, err := Compile(output)
		if err != nil {
			t.Fatal(err)
		}
		runtime := compiled.NewRuntimeAttributes()
		next := output.Attrs.(tensor.QSAAttentionAttributes)
		next.Cells = []uint32{7, 8}
		if err := runtime.Set(output, next); err != nil {
			t.Fatal(err)
		}
		if runtime.launchArguments == inPlace {
			t.Fatalf("in place %t: the step's next cells reach its kernels as launch arguments: %t", inPlace, runtime.launchArguments)
		}
	}
}

// qsaStateShape is one state tensor's name, type and shape, in tensor.QSAState's order (the operation's inputs from
// its key codes).
type qsaStateShape struct {
	name  string
	kind  dtype.Type
	shape tensor.Shape
}

func qsaStateShapes(capacity uint64) []qsaStateShape {
	pages := capacity / tensor.QSABlockCells
	return []qsaStateShape{
		{"key_codes", dtype.I8, tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*capacity)},
		{"value_codes", dtype.I8, tensor.MustShape(tensor.QSAHeadWidth, tensor.QSAKeyValueHeads*capacity)},
		{"key_scales", dtype.F16, tensor.MustShape(tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*capacity)},
		{"value_scales", dtype.F16, tensor.MustShape(tensor.QSAHeadWidth/tensor.QSAScaleGroup, tensor.QSAKeyValueHeads*capacity)},
		{"page_table", dtype.I32, tensor.MustShape(pages)},
		{"indexer_pooled", dtype.F32, tensor.MustShape(tensor.QSAIndexerWidth, pages+tensor.QSAPooledSpare)},
		{"block_position", dtype.I32, tensor.MustShape(1)},
		{"positions", dtype.I32, tensor.MustShape(tensor.QSAPositionAxes, capacity)},
		{"rope_cos", dtype.F32, tensor.MustShape(tensor.QSARotated/2, capacity)},
		{"rope_sin", dtype.F32, tensor.MustShape(tensor.QSARotated/2, capacity)},
	}
}
