//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestStrataNativeMMVQMatchesStrata holds Strata's native MMVQ projections (strata_native_mmvq.cuh, from
// native_mmvq.cu) to the checks Strata keeps beside them, over Flash-Next's dense formats and shapes: the q8_1 quantizer
// gives each 32-value block the scale of its largest magnitude over 127 and codes within half a step of the values
// (q8_1_finite.hpp); a projection, Flash-Next's formats' and the K-quants' Overgo routes through them (Q4_K, Q5_K), is a
// float64 transcription's over the weights and the quantized activations within
// float rounding, on the single-column kernel, its small-K form, and every column count of the multi-column kernel; every
// column of a multi-column call is bitwise the single-column call on that column (bench/micro/native_mmvq_multi.cpp);
// and Q6_K's interleaved kernels at two to four columns and one, two and four rows a warp are bitwise the multi-column
// kernel's (mmvq_il_parity).
func TestStrataNativeMMVQMatchesStrata(t *testing.T) {
	const (
		columns = strataNativeMaxColumns
		// bound is the relative L1 distance a projection keeps from the float64 transcription.
		bound = 1e-5
		// weightScale and activationScale size the weights and activations as Flash-Next's run.
		weightScale     = 0.02
		activationScale = 1.0
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(19, 23))
	type shape struct {
		name      string
		ggml      int
		storage   dtype.Type
		nIn, nOut int32
	}
	for _, s := range []shape{
		{"q8_0 qkv", strataGGMLQ8_0, dtype.Q8_0, 2560, 1040},
		{"q8_0 small-K down", strataGGMLQ8_0, dtype.Q8_0, 640, 520},
		{"q6_K head", strataGGMLQ6K, dtype.Q6K, 2560, 2100},
		{"q6_K small-K", strataGGMLQ6K, dtype.Q6K, 768, 300},
		{"q5_K projection", strataGGMLQ5K, dtype.Q5K, 5120, 1040},
		{"q5_K small-K", strataGGMLQ5K, dtype.Q5K, 1536, 300},
		{"q4_K projection", strataGGMLQ4K, dtype.Q4K, 5120, 1040},
		{"q4_K small-K", strataGGMLQ4K, dtype.Q4K, 1536, 300},
	} {
		weights := strataRandomWeights(t, draws, s.storage, s.nIn, s.nOut, weightScale)
		reference, err := quant.Dequantize(s.storage, weights, uint64(s.nIn)*uint64(s.nOut))
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, int(s.nIn)*columns)
		for index := range x {
			x[index] = float32(draws.NormFloat64()) * activationScale
		}
		weightsBuffer, xBuffer := buffers.upload(t, weights), buffers.upload(t, float32Bytes(x))
		xq := buffers.zeros(t, int(strataNativeQ81Size(s.nIn, columns)))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataNativeQuantizeQ81(state, functions, xBuffer, xq, s.nIn, columns)
		})
		blocks := buffers.read(t, xq, int(strataNativeQ81Size(s.nIn, columns)))
		dequantized := strataQ81Values(t, blocks, x)
		project := func(ncols int32, first int) []float32 {
			y := buffers.zeros(t, 4*int(s.nOut)*int(ncols))
			at := xq + driver.DevicePtr(strataNativeQ81Size(s.nIn, int32(first)))
			buffers.run(t, func(state *device.State, functions functionSet) error {
				return launchStrataNativeMMVQ(state, functions, s.ggml, weightsBuffer, at, y, s.nIn, s.nOut, ncols)
			})
			return float32s(buffers.read(t, y, 4*int(s.nOut)*int(ncols)))
		}
		singles := make([][]float32, columns)
		for column := range columns {
			singles[column] = project(1, column)
			want := make([]float64, s.nOut)
			for row := range int(s.nOut) {
				var sum float64
				for k := range int(s.nIn) {
					sum += float64(reference[row*int(s.nIn)+k]) * dequantized[column*int(s.nIn)+k]
				}
				want[row] = sum
			}
			if d := strataRelativeL1(float32To64(singles[column]), want); d > bound {
				t.Fatalf("%s: column %d is %.3g from the transcription", s.name, column, d)
			}
		}
		multis := make(map[int32][]float32)
		for ncols := int32(strataNativeMultiFirst); ncols <= columns; ncols++ {
			got := project(ncols, 0)
			multis[ncols] = got
			for column := range int(ncols) {
				if !slices.Equal(got[column*int(s.nOut):(column+1)*int(s.nOut)], singles[column]) {
					t.Fatalf("%s: column %d of a %d-column call differs from its single-column call", s.name, column, ncols)
				}
			}
		}
		if s.ggml != strataGGMLQ6K {
			continue
		}
		for ncols := int32(strataNativeMultiFirst); ncols <= strataNativeILMaxColumns; ncols++ {
			xil := buffers.zeros(t, int(strataNativeQ81ILSize(s.nIn, ncols)))
			buffers.run(t, func(state *device.State, functions functionSet) error {
				return launchStrataNativeQ81Interleave(state, functions, xq, xil, s.nIn, ncols)
			})
			for rowIndex, kernel := range strataNativeILQ6K[ncols-strataNativeMultiFirst] {
				y := buffers.zeros(t, 4*int(s.nOut)*int(ncols))
				bm, _, scales := strataNativeQ81ILParts(xil, s.nIn, ncols)
				rows := [...]uint32{1, 2, 4}[rowIndex]
				blocks := ((uint32(s.nOut)+rows-1)/rows + strataNativeILWarpsABlock - 1) / strataNativeILWarpsABlock
				nIn, nOut := s.nIn, s.nOut
				buffers.run(t, func(state *device.State, functions functionSet) error {
					return launchGridABI(state, functions[kernel], driver.Dim3{X: blocks, Y: 1, Z: 1},
						driver.Dim3{X: strataNativeILWarpsABlock * strataNativeWarp, Y: 1, Z: 1},
						&weightsBuffer, &bm, &scales, &y, &nIn, &nOut)
				})
				if got := float32s(buffers.read(t, y, 4*int(s.nOut)*int(ncols))); !slices.Equal(got, multis[ncols]) {
					t.Fatalf("%s: the interleaved kernel at %d columns, %d rows a warp, differs from the multi-column kernel", s.name, ncols, rows)
				}
			}
		}
	}
}

// strataRandomWeights draws a matrix of n_out rows of n_in values in storage's blocks: Q8_0's codes and fp16 scales,
// Q6_K's codes, its signed 6-bit sub-block scales and its fp16 scale, Q4_K's and Q5_K's codes, packed 6-bit scales and
// mins and their fp16 scale and min, each of the given magnitude.
func strataRandomWeights(t *testing.T, draws *rand.Rand, storage dtype.Type, nIn, nOut int32, scale float32) []byte {
	t.Helper()
	traits, _ := storage.Traits()
	blocks := uint64(nIn) / traits.BlockSize * uint64(nOut)
	data := make([]byte, blocks*traits.TypeSize)
	for index := range data {
		data[index] = byte(draws.Uint32())
	}
	for block := range blocks {
		at := block * traits.TypeSize
		switch storage {
		case dtype.Q8_0:
			binary.LittleEndian.PutUint16(data[at:], quant.Float32ToFloat16(scale*draws.Float32()/127))
		case dtype.Q6K:
			for index := range uint64(16) {
				data[at+192+index] = byte(int8(draws.IntN(63) - 31))
			}
			binary.LittleEndian.PutUint16(data[at+208:], quant.Float32ToFloat16(scale*draws.Float32()/(31*32)))
		case dtype.Q4K, dtype.Q5K:
			binary.LittleEndian.PutUint16(data[at:], quant.Float32ToFloat16(scale*draws.Float32()/(63*16)))
			binary.LittleEndian.PutUint16(data[at+2:], quant.Float32ToFloat16(scale*draws.Float32()/(63*16)))
		}
	}
	return data
}

// strataQ81Values reads q8_1 blocks back to their values (code times the block's fp16 scale), holding each block's
// scale to its values' largest magnitude over 127 and each code to within half a step of its value.
func strataQ81Values(t *testing.T, blocks []byte, x []float32) []float64 {
	t.Helper()
	const step = 127
	values := make([]float64, len(x))
	for block := range len(x) / strataNativeQ8K {
		at := block * strataNativeQ81Bytes
		d := float64(quant.Float16ToFloat32(binary.LittleEndian.Uint16(blocks[at:])))
		var amax float64
		for index := range strataNativeQ8K {
			amax = max(amax, math.Abs(float64(x[block*strataNativeQ8K+index])))
		}
		if math.Abs(d-amax/step) > amax/step*1e-3 {
			t.Fatalf("q8_1 block %d's scale %g is not its largest magnitude %g over 127", block, d, amax)
		}
		// The codes are of the unrounded scale; the block stores it as fp16.
		unrounded := float64(float32(amax) / step)
		for index := range strataNativeQ8K {
			code := float64(int8(blocks[at+4+index]))
			value := float64(x[block*strataNativeQ8K+index])
			if unrounded > 0 && math.Abs(code-value/unrounded) > 0.5+1e-3 {
				t.Fatalf("q8_1 block %d's code %d is more than half a step from %g", block, int(code), value/unrounded)
			}
			values[block*strataNativeQ8K+index] = code * d
		}
	}
	return values
}
