//go:build windows

package executor

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// TestStrataCellAttentionGeometries holds Strata's int8 cell attention, instantiated for an attention layer's own
// geometry (widen-kv-kernels), to Strata's parity program for its own (qsa_prompt_attn_parity.cpp, as
// strataAttentionParity transcribes it): random codes and scales behind a shuffled page table, the decode attention
// within 1e-3 of the output scale of an FP64 attention over the dequantized cells, the prompt attention within four
// times the decode attention's error and within 1e-4 of the output scale of it. Each geometry is read three ways: a
// selection, as Strata's QSA layer reads, a causal prefix and a sliding window, both a range of cells its step record
// names. The geometries are gemma-4-12B's sliding layers (8 K/V heads of 256, 2 query heads each, a 1,024-token
// window), its global layers (one K/V head of 512, 16 query heads) and a llama layer's (2 K/V heads of 128, 8 each);
// each instantiation's shared memory is the compiler's size of its layout.
func TestStrataCellAttentionGeometries(t *testing.T) {
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	const window = 1024
	for _, c := range []struct {
		name     string
		geometry strataCellGeometry
	}{
		{"gemma-4-12B sliding", strataCellGeometry{width: 256, group: 2, kvHeads: 8, pageCells: 16}},
		{"gemma-4-12B global", strataCellGeometry{width: 512, group: 16, kvHeads: 1, pageCells: 16}},
		{"llama", strataCellGeometry{width: 128, group: 8, kvHeads: 2, pageCells: 4}},
	} {
		for _, mode := range []string{"selection", "causal", "window"} {
			t.Run(c.name+"/"+mode, func(t *testing.T) { strataCellParity(t, cuda, c.geometry, mode, window) })
		}
	}
}

func strataCellParity(t *testing.T, cuda *Executor, geometry strataCellGeometry, mode string, window int64) {
	const cells, queries = 2500, 96
	width, heads := geometry.width, geometry.kvHeads*geometry.group
	pages := (cells + geometry.pageCells - 1) / geometry.pageCells
	rows := pages * geometry.kvHeads * geometry.pageCells
	groups := width / int64(strataKVQ8Group)
	draws := rand.New(rand.NewPCG(uint64(width), uint64(geometry.group)))
	keyCodes, valueCodes := make([]int8, rows*width), make([]int8, rows*width)
	for index := range keyCodes {
		keyCodes[index], valueCodes[index] = int8(draws.IntN(255)-127), int8(draws.IntN(255)-127)
	}
	keyScales, valueScales := make([]uint16, rows*groups), make([]uint16, rows*groups)
	for index := range keyScales {
		keyScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
		valueScales[index] = strataF16FromF32(0.005 + 0.025*draws.Float32())
	}
	table := make([]int32, pages)
	for index, page := range draws.Perm(int(pages)) {
		table[index] = int32(page)
	}
	// Each query at its position reads its cells: a selection of a recent stretch and older cells, its causal prefix,
	// or its sliding window; the capacity is the most any query reads.
	read := func(position int64) (first, count int64) {
		if mode == "window" {
			first = max(0, position+1-window)
		}
		return first, position + 1 - first
	}
	capacity := int64(0)
	for query := range int64(queries) {
		_, count := read(cells - queries + query)
		capacity = max(capacity, min(count, 700))
	}
	if mode != "selection" {
		capacity = 0
		for query := range int64(queries) {
			_, count := read(cells - queries + query)
			capacity = max(capacity, count)
		}
	}
	selections := make([][]int32, queries)
	steps := make([]int32, queries*strataStepCount)
	var ids []int32
	if mode == "selection" {
		ids = make([]int32, queries*capacity)
	}
	for query := range int64(queries) {
		position := cells - queries + query
		first, count := read(position)
		steps[query*strataStepCount] = int32(position)
		steps[query*strataStepCount+1] = int32(position + 1)
		if mode == "selection" {
			chosen := draws.Perm(int(count))[:min(count, capacity)]
			selection := make([]int32, len(chosen))
			for index, cell := range chosen {
				selection[index] = int32(cell)
			}
			selections[query] = selection
			copy(ids[query*capacity:], selection)
			steps[query*strataStepCount+3] = int32(len(selection))
			continue
		}
		selection := make([]int32, count)
		for index := range selection {
			selection[index] = int32(first + int64(index))
		}
		selections[query] = selection
		steps[query*strataStepCount+2] = int32(first)
		steps[query*strataStepCount+3] = int32(count)
	}
	q := make([]float32, queries*heads*width)
	for index := range q {
		q[index] = float32(draws.NormFloat64()) * 2
	}

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	pools := strataAttentionPools{
		kv: strataKVPools{
			key: buffers.upload(t, int8Bytes(keyCodes)), value: buffers.upload(t, int8Bytes(valueCodes)),
			keyScale: buffers.upload(t, uint16Bytes(keyScales)), valueScale: buffers.upload(t, uint16Bytes(valueScales)),
		},
		table: buffers.upload(t, int32Bytes(table)),
	}
	var idBuffer driver.DevicePtr
	if ids != nil {
		idBuffer = buffers.upload(t, int32Bytes(ids))
	}
	stepBuffer, queryBuffer := buffers.upload(t, int32Bytes(steps)), buffers.upload(t, float32Bytes(q))
	outputBytes := int(queries * heads * width * 4)
	chunks := (capacity + strataAttentionChunk - 1) / strataAttentionChunk
	decodeOutput, promptOutput := buffers.zeros(t, outputBytes), buffers.zeros(t, outputBytes)
	scratch := buffers.zeros(t, int(queries*chunks*heads*(width+2)*4))
	kernels, _ := strataCellKernelsOf(geometry)
	sizeBuffer := buffers.zeros(t, 4)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchGridABI(state, functions[kernels.sharedBytes], driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, &sizeBuffer)
	})
	if sharedBytes, want := binary.LittleEndian.Uint32(buffers.read(t, sizeBuffer, 4)), strataCellPromptSharedBytes(width); sharedBytes != want {
		t.Fatalf("the prompt kernel's shared layout is %d bytes, the launch gives %d", sharedBytes, want)
	}
	scoreScale := float32(1 / math.Sqrt(float64(width)))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataCellDecode(state, functions, geometry, queryBuffer, pools, idBuffer, stepBuffer, capacity, scoreScale,
			scratch, decodeOutput, queries); err != nil {
			return err
		}
		return launchStrataCellPrompt(state, functions, geometry, queryBuffer, pools, idBuffer, stepBuffer, capacity, scoreScale,
			promptOutput, queries)
	})
	decode := float32s(buffers.read(t, decodeOutput, outputBytes))
	prompt := float32s(buffers.read(t, promptOutput, outputBytes))
	value := func(codes []int8, scales []uint16, row, d int64) float64 {
		return float64(codes[row*width+d]) * float64(strataF32FromF16(scales[row*groups+d/int64(strataKVQ8Group)]))
	}
	errorDecode, errorPrompt, referenceScale := 0.0, 0.0, 0.0
	for query := int64(0); query < queries; query += max(1, queries/16) {
		selection := selections[query]
		for head := range heads {
			kvHead := head / geometry.group
			rowOf := func(cell int32) int64 {
				return (int64(table[int64(cell)/geometry.pageCells])*geometry.kvHeads+kvHead)*geometry.pageCells + int64(cell)%geometry.pageCells
			}
			scores := make([]float64, len(selection))
			maximum := math.Inf(-1)
			for index, cell := range selection {
				row, dot := rowOf(cell), 0.0
				for d := range width {
					dot += float64(q[(query*heads+head)*width+d]) * value(keyCodes, keyScales, row, d)
				}
				scores[index] = dot / math.Sqrt(float64(width))
				maximum = max(maximum, scores[index])
			}
			sum := 0.0
			for index := range scores {
				scores[index] = math.Exp(scores[index] - maximum)
				sum += scores[index]
			}
			for d := range width {
				accumulated := 0.0
				for index, cell := range selection {
					accumulated += scores[index] * value(valueCodes, valueScales, rowOf(cell), d)
				}
				reference := accumulated / sum
				at := (query*heads+head)*width + d
				referenceScale = max(referenceScale, math.Abs(reference))
				errorDecode = max(errorDecode, math.Abs(float64(decode[at])-reference))
				errorPrompt = max(errorPrompt, math.Abs(float64(prompt[at])-reference))
			}
		}
	}
	difference, scale := 0.0, 0.0
	for index := range decode {
		difference = max(difference, math.Abs(float64(decode[index]-prompt[index])))
		scale = max(scale, math.Abs(float64(decode[index])))
	}
	if !(errorDecode <= 1e-3*referenceScale) {
		t.Errorf("the decode attention stands %.3g from FP64, past 1e-3 of the output scale %.3g", errorDecode, referenceScale)
	}
	if !(errorPrompt <= max(4*errorDecode, 1e-6*referenceScale)) {
		t.Errorf("the prompt attention stands %.3g from FP64, past four times the decode attention's %.3g", errorPrompt, errorDecode)
	}
	if !(difference <= 1e-4*scale) {
		t.Errorf("the prompt and decode attention differ by %.3g, past 1e-4 of the output scale %.3g", difference, scale)
	}
	t.Logf("%s: decode %.3g, prompt %.3g from FP64 (scale %.3g); the two %.2g of the scale",
		fmt.Sprint(geometry), errorDecode, errorPrompt, referenceScale, difference/scale)
}
