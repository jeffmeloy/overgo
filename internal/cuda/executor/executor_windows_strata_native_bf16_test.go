//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// TestStrataBF16GEMVMatchesStrata holds Strata's native bf16 projections (strata_native_bf16.cuh, from native_bf16.cu)
// to their contract at Flash-Next's widths (the router's 512 outputs, the indexer's 128, the shared expert's gate input
// of one): every row of a multi-row call, one to eight rows, is bitwise its one-row call, and each output is a float64
// transcription's over the bf16 weights within float rounding.
func TestStrataBF16GEMVMatchesStrata(t *testing.T) {
	const (
		nIn = 2560
		// bound is the relative L1 distance a projection keeps from the float64 transcription.
		bound = 1e-6
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(37, 41))
	for _, nOut := range []int{512, 128, 1} {
		weights := make([]byte, 2*nIn*nOut)
		values := make([]float64, nIn*nOut)
		for index := range values {
			bits := uint16(math.Float32bits(float32(draws.NormFloat64())*0.05) >> 16)
			weights[2*index], weights[2*index+1] = byte(bits), byte(bits>>8)
			values[index] = float64(math.Float32frombits(uint32(bits) << 16))
		}
		x := make([]float32, strataBF16MaxRows*nIn)
		for index := range x {
			x[index] = float32(draws.NormFloat64())
		}
		weightBuffer, xBuffer := buffers.upload(t, weights), buffers.upload(t, float32Bytes(x))
		singles := make([][]float32, strataBF16MaxRows)
		for row := range strataBF16MaxRows {
			y := buffers.zeros(t, 4*nOut)
			at := xBuffer + driver.DevicePtr(4*row*nIn)
			buffers.run(t, func(state *device.State, functions functionSet) error {
				return launchStrataBF16GEMV(state, functions, at, weightBuffer, y, nIn, int64(nOut))
			})
			singles[row] = float32s(buffers.read(t, y, 4*nOut))
			want := make([]float64, nOut)
			for out := range nOut {
				for k := range nIn {
					want[out] += values[out*nIn+k] * float64(x[row*nIn+k])
				}
			}
			if d := strataRelativeL1(float32To64(singles[row]), want); d > bound {
				t.Fatalf("%d outputs: row %d is %.3g from the transcription", nOut, row, d)
			}
		}
		for rows := int32(1); rows <= strataBF16MaxRows; rows++ {
			// ldy past n_out keeps a one-row call on the multi-row kernel.
			ldy := int64(nOut) + 1
			y := buffers.zeros(t, 4*int(ldy)*int(rows))
			buffers.run(t, func(state *device.State, functions functionSet) error {
				return launchStrataBF16GEMVMulti(state, functions, xBuffer, nIn, weightBuffer, y, ldy, nIn, int64(nOut), rows)
			})
			got := float32s(buffers.read(t, y, 4*int(ldy)*int(rows)))
			for row := range int(rows) {
				if !slices.Equal(got[row*int(ldy):row*int(ldy)+nOut], singles[row]) {
					t.Fatalf("%d outputs: row %d of a %d-row call differs from its one-row call", nOut, row, rows)
				}
			}
		}
	}
}
