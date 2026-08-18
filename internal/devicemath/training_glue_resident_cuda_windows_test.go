//go:build windows

package devicemath

import (
	"errors"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

func TestResidentTrainingGlueMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointers []driver.DevicePtr
	allocate := func(count int, initial []float32) driver.DevicePtr {
		t.Helper()
		pointer, err := AllocResidentF32(worker, count, initial)
		if err != nil {
			t.Fatal(err)
		}
		pointers = append(pointers, pointer)
		return pointer
	}
	allocateRows := func(rows []uint32) driver.DevicePtr {
		t.Helper()
		pointer, err := AllocResidentU32(worker, rows)
		if err != nil {
			t.Fatal(err)
		}
		pointers = append(pointers, pointer)
		return pointer
	}
	read := func(pointer driver.DevicePtr, count int) []float32 {
		t.Helper()
		values := make([]float32, count)
		if err := ReadResident(worker, pointer, ResidentSlice{Data: values}); err != nil {
			t.Fatal(err)
		}
		return values
	}
	defer func() {
		if err := FreeResident(worker, pointers...); err != nil {
			t.Error(err)
		}
	}()

	left := []float32{-2, -1, 0, 1, 2, 3}
	right := []float32{3, 2, 1, 0, -1, -2}
	incoming := []float32{1, 2, 3, 4, 5, 6}
	leftPtr, rightPtr := allocate(len(left), left), allocate(len(right), right)
	addPtr, reluPtr := allocate(len(left), nil), allocate(len(left), nil)
	if err := AddResident(worker, leftPtr, rightPtr, addPtr, len(left)); err != nil {
		t.Fatal(err)
	}
	if err := ScaleResident(worker, addPtr, addPtr, 2.5, len(left)); err != nil {
		t.Fatal(err)
	}
	scalarPtr := allocate(1, []float32{0.4})
	if err := ScaleByResidentScalar(worker, addPtr, scalarPtr, addPtr, len(left)); err != nil {
		t.Fatal(err)
	}
	incomingPtr := allocate(len(incoming), incoming)
	if err := ReLUBackwardResident(worker, incomingPtr, leftPtr, reluPtr, len(left)); err != nil {
		t.Fatal(err)
	}
	if delta := testutil.MaxAbsDiff(read(addPtr, len(left)), []float32{1, 1, 1, 1, 1, 1}); delta != 0 {
		t.Fatalf("resident add delta %.3e", delta)
	}
	if delta := testutil.MaxAbsDiff(read(reluPtr, len(left)), []float32{0, 0, 0, 4, 5, 6}); delta != 0 {
		t.Fatalf("resident ReLU VJP delta %.3e", delta)
	}

	stridedSource := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	stridedPtr, copiedPtr := allocate(len(stridedSource), stridedSource), allocate(12, nil)
	if err := StridedRowCopyResident(worker, stridedPtr, copiedPtr, 3, 2, 5, 1, 4, 1); err != nil {
		t.Fatal(err)
	}
	if delta := testutil.MaxAbsDiff(read(copiedPtr, 12), []float32{0, 1, 2, 0, 0, 6, 7, 0, 0, 11, 12, 0}); delta != 0 {
		t.Fatalf("resident strided copy delta %.3e", delta)
	}

	scatterSource := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	rowIDs := []uint32{2, 1, 2, 0}
	scatterSourcePtr, rowIDsPtr, tablePtr := allocate(len(scatterSource), scatterSource), allocateRows(rowIDs), allocate(9, nil)
	if err := IndexedRowScatterAddResident(worker, scatterSourcePtr, rowIDsPtr, tablePtr, len(rowIDs), 3); err != nil {
		t.Fatal(err)
	}
	if delta := testutil.MaxAbsDiff(read(tablePtr, 9), []float32{10, 11, 12, 4, 5, 6, 8, 10, 12}); delta != 0 {
		t.Fatalf("resident indexed scatter delta %.3e", delta)
	}

	const sequence, headDim, lagCount = 4, 3, 4
	rng := rand.New(rand.NewSource(59))
	query, key := randSlice(rng, sequence*headDim), randSlice(rng, sequence*headDim)
	dScores := make([]float32, sequence*sequence)
	for queryRow := range sequence {
		for keyRow := 0; keyRow <= queryRow; keyRow++ {
			dScores[queryRow*sequence+keyRow] = float32(rng.NormFloat64())
		}
	}
	wantBias, wantScale := make([]float32, lagCount), float32(0)
	for queryRow := range sequence {
		for keyRow := 0; keyRow <= queryRow; keyRow++ {
			gradient := dScores[queryRow*sequence+keyRow]
			wantBias[queryRow-keyRow] += gradient
			var dot float32
			for channel := range headDim {
				dot += query[queryRow*headDim+channel] * key[keyRow*headDim+channel]
			}
			wantScale += gradient * dot
		}
	}
	dScoresPtr, queryPtr, keyPtr := allocate(len(dScores), dScores), allocate(len(query), query), allocate(len(key), key)
	biasPtr, scalePtr := allocate(lagCount, nil), allocate(1, nil)
	if err := AttentionScoreAffineBackwardResident(worker, dScoresPtr, queryPtr, keyPtr, biasPtr, scalePtr, sequence, headDim, lagCount); err != nil {
		t.Fatal(err)
	}
	biasDelta, scaleDelta := testutil.MaxAbsDiff(read(biasPtr, lagCount), wantBias), testutil.MaxAbsDiff(read(scalePtr, 1), []float32{wantScale})
	t.Logf("resident glue bias=%.3e scale=%.3e", biasDelta, scaleDelta)
	if biasDelta > 5e-7 || scaleDelta > 5e-7 {
		t.Fatalf("resident attention reduction differs: bias=%.3e scale=%.3e", biasDelta, scaleDelta)
	}
}

func TestResidentOpsSessionMatchesWrappers(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	left, right := []float32{-2, -1, 0, 1, 2, 3}, []float32{3, 2, 1, 0, -1, -2}
	leftPtr, err := AllocResidentF32(worker, len(left), left)
	if err != nil {
		t.Fatal(err)
	}
	rightPtr, err := AllocResidentF32(worker, len(right), right)
	if err != nil {
		_ = FreeResident(worker, leftPtr)
		t.Fatal(err)
	}
	wrapperPtr, err := AllocResidentF32(worker, len(left), nil)
	if err != nil {
		_ = FreeResident(worker, leftPtr, rightPtr)
		t.Fatal(err)
	}
	sessionPtr, err := AllocResidentF32(worker, len(left), nil)
	if err != nil {
		_ = FreeResident(worker, leftPtr, rightPtr, wrapperPtr)
		t.Fatal(err)
	}
	defer FreeResident(worker, leftPtr, rightPtr, wrapperPtr, sessionPtr)
	if err := AddResident(worker, leftPtr, rightPtr, wrapperPtr, len(left)); err != nil {
		t.Fatal(err)
	}
	if err := WithResidentOps(worker, func(ops *ResidentOps) error {
		if err := ops.Add(leftPtr, rightPtr, sessionPtr, len(left)); err != nil {
			return err
		}
		return ops.Scale(sessionPtr, sessionPtr, 1, len(left))
	}); err != nil {
		t.Fatal(err)
	}
	wrapper, session := make([]float32, len(left)), make([]float32, len(left))
	if err := ReadResident(worker, wrapperPtr, ResidentSlice{Data: wrapper}); err != nil {
		t.Fatal(err)
	}
	if err := ReadResident(worker, sessionPtr, ResidentSlice{Data: session}); err != nil {
		t.Fatal(err)
	}
	delta := testutil.MaxAbsDiff(wrapper, session)
	t.Logf("resident session/wrappers delta=%.3e", delta)
	if delta != 0 {
		t.Fatalf("resident session differs: %.3e", delta)
	}
}

func TestResidentArenaReusesStorage(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	if err := WithResidentOps(worker, func(ops *ResidentOps) error {
		arena, err := ops.NewArena(12)
		if err != nil {
			return err
		}
		if _, err := arena.AllocF32(4); err != nil {
			return err
		}
		mark := arena.Mark()
		first, err := arena.AllocF32(8)
		if err != nil {
			return err
		}
		if _, err := arena.AllocF32(1); err == nil {
			return errors.New("resident arena accepted overflow")
		}
		if err := arena.Reset(mark); err != nil {
			return err
		}
		second, err := arena.AllocF32(8)
		if err != nil {
			return err
		}
		if first != second {
			return errors.New("resident arena rewind changed address")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
