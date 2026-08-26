//go:build windows

package controllertrain_test

import (
	"errors"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/controllertrain"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/devicemath"
	"overgo/internal/scratchmodel"
	"overgo/internal/scratchmodeltest"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestTier1TrainingSchedule(t *testing.T) {
	cudatest.Require(t)
	commit := controllerSourceCommit(t)
	corpus, err := controllertrain.Compile(controllertrain.Spec{
		Train: controllerRecords(commit, false), Holdout: controllerRecords(commit, true),
	})
	if err != nil {
		t.Fatal(err)
	}
	construction, err := scratchmodel.Compile(
		scratchmodel.CorpusFacts{Documents: corpus.TrainingDocuments(), Seed: 17, Steps: 800},
		scratchmodeltest.Profile(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	measurementWorker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	granularity, measureErr := devicemath.MeasureAllocGranularity(measurementWorker)
	closeErr := measurementWorker.Close()
	if err := errors.Join(measureErr, closeErr); err != nil {
		t.Fatal(err)
	}
	segments, err := construction.TrainingMemorySegments()
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := scratchmodel.NewResidentTrainer(construction, 800, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if err := trainer.ResetPeakMemory(); err != nil {
		t.Fatal(err)
	}
	tokens, err := construction.Tokens(construction.Split().Train[0])
	if err != nil {
		t.Fatal(err)
	}
	loss, err := trainer.Step(tokens, 1)
	if err != nil || math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("measured controller step = %g, %v", loss, err)
	}
	memory, err := trainer.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	calibration := testutil.ArtifactID(t, artifact.KindEvidence, "controller-tier1-live-cuda-measurement")
	resident, err := trainingprogram.CompileMemorySchedule(trainingprogram.MemoryScheduleSpec{
		Calibration: calibration, AllocationGranularity: granularity,
		MeasuredResidentPeak: memory.PeakBytes, DeviceBudget: ^uint64(0), HostBudget: ^uint64(0),
		Segments: segments,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resident.Tier1PeakBytes() >= resident.ResidentPeakBytes() {
		t.Fatalf("controller has no logical Tier-1 reduction: resident=%d tier1=%d",
			resident.ResidentPeakBytes(), resident.Tier1PeakBytes())
	}
	forced, err := trainingprogram.CompileMemorySchedule(trainingprogram.MemoryScheduleSpec{
		Calibration: calibration, AllocationGranularity: granularity,
		MeasuredResidentPeak: memory.PeakBytes, DeviceBudget: resident.Tier1PeakBytes(),
		HostBudget: ^uint64(0), Segments: segments,
	})
	if err != nil {
		t.Fatal(err)
	}
	if forced.Tier() != trainingprogram.MemoryTierHost || !forced.DoubleBuffered() ||
		forced.ConnectorDeviceBytes() != 0 || forced.UsesActivationCheckpointing() ||
		forced.UsesLayerMajorAccumulation() || forced.UsesOptimizerPaging() {
		t.Fatalf("forced Tier-1 policy = tier=%s double=%v connector=%d checkpoint=%v layer-major=%v paging=%v",
			forced.Tier(), forced.DoubleBuffered(), forced.ConnectorDeviceBytes(),
			forced.UsesActivationCheckpointing(), forced.UsesLayerMajorAccumulation(), forced.UsesOptimizerPaging())
	}
	if forced.HostBytes() == 0 || forced.FixedDeviceBytes() == 0 || forced.Calibration() != calibration {
		t.Fatalf("forced Tier-1 evidence = host=%d fixed=%d calibration=%s",
			forced.HostBytes(), forced.FixedDeviceBytes(), forced.Calibration())
	}
	t.Logf("controller Tier-1 schedule: loss=%.6f logical resident=%d forced=%d physical resident=%d forced=%d granularity=%d host=%d segments=%d; no connector state applicable",
		loss, forced.ResidentPeakBytes(), forced.Tier1PeakBytes(),
		forced.ResidentCapacityBytes(), forced.Tier1CapacityBytes(), granularity,
		forced.HostBytes(), len(forced.Segments()))
}
