//go:build windows

package devicemath

import (
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestTheCardsGradientSumsRepeatBitForBit runs, 20 times on the same inputs, the two sums a training step on the card
// makes of rows that meet in one place (fixed-order/do): RMSNorm's weight gradient over 300 rows, two blocks of them,
// and the embedding gradient's scatter of 600 positions onto the rows of 7 tokens. Each gives the same bytes every time.
// Summed by atomic adds, the norm's gradient differed from the first run's in 49 of 49 repeats over 300 rows, and a
// reward session stopped and resumed on the card ended with other bytes than the uninterrupted one in three of six
// runs of TestDeviceSamplerDrawsWhatTheHostSamplerDraws.
func TestTheCardsGradientSumsRepeatBitForBit(t *testing.T) {
	cudatest.Require(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	rng := rand.New(rand.NewSource(61))
	const rows, width, repeats = 300, 64, 20
	x, weight, dy := randSlice(rng, rows*width), randSlice(rng, width), randSlice(rng, rows*width)
	var normFirst []float32
	for repeat := range repeats {
		_, dscale, err := RMSNormBackward(worker, x, weight, dy, rows, width, 1e-6)
		testcheck.NoError(t, err)
		if normFirst == nil {
			normFirst = dscale
		} else if !slices.Equal(dscale, normFirst) {
			t.Fatalf("run %d of the norm's weight gradient over %d rows has other bytes than the first", repeat+1, rows)
		}
	}

	const positions, tokens = 600, 7
	var pointers []driver.DevicePtr
	defer func() {
		if err := FreeResident(worker, pointers...); err != nil {
			t.Error(err)
		}
	}()
	allocate := residentAllocation(t, worker, &pointers)
	ids := make([]uint32, positions)
	for position := range ids {
		ids[position] = uint32(rng.Intn(tokens))
	}
	idsPtr := testcheck.Must(AllocResidentU32(worker, ids))(t)
	pointers = append(pointers, idsPtr)
	sourcePtr := allocate(positions*width, randSlice(rng, positions*width))
	var scatterFirst []float32
	for repeat := range repeats {
		table := allocate(tokens*width, make([]float32, tokens*width))
		testcheck.NoError(t, IndexedRowScatterAddResident(worker, sourcePtr, idsPtr, table, positions, width))
		summed := make([]float32, tokens*width)
		testcheck.NoError(t, ReadResident(worker, table, ResidentSlice{Data: summed}))
		if scatterFirst == nil {
			scatterFirst = summed
		} else if !slices.Equal(summed, scatterFirst) {
			t.Fatalf("run %d of the scatter of %d positions onto %d tokens has other bytes than the first", repeat+1, positions, tokens)
		}
	}
}
