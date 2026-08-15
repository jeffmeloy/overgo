package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMemoryScheduleRequiresMeasuredFit(t *testing.T) {
	segments := []MemorySegment{
		{Name: "retained", ParameterBytes: 8, GradientBytes: 8, OptimizerBytes: 8, Retained: true},
		{Name: "layer-0", ParameterBytes: 80, GradientBytes: 80, OptimizerBytes: 80},
		{Name: "layer-1", ParameterBytes: 64, GradientBytes: 64, OptimizerBytes: 64},
	}
	base := MemoryScheduleSpec{
		Calibration:           testutil.ArtifactID(t, artifact.KindEvidence, "memory-calibration"),
		AllocationGranularity: 64, MeasuredResidentPeak: 1024,
		DeviceBudget: 1024, HostBudget: 1024, Segments: segments,
	}
	resident, err := CompileMemorySchedule(base)
	if err != nil || resident.Tier() != MemoryTierDevice || resident.Tier1PeakBytes() >= resident.ResidentPeakBytes() {
		t.Fatalf("resident schedule = (%s, %d/%d, %v)",
			resident.Tier(), resident.Tier1PeakBytes(), resident.ResidentPeakBytes(), err)
	}
	base.DeviceBudget = resident.Tier1PeakBytes()
	streamed, err := CompileMemorySchedule(base)
	if err != nil || streamed.Tier() != MemoryTierHost || !streamed.DoubleBuffered() ||
		streamed.UsesActivationCheckpointing() || streamed.UsesLayerMajorAccumulation() || streamed.UsesOptimizerPaging() {
		t.Fatalf("streamed schedule = (%s, %v, %v)", streamed.Tier(), streamed.DoubleBuffered(), err)
	}
	base.DeviceBudget--
	if _, err := CompileMemorySchedule(base); err == nil {
		t.Fatal("unmeasured optional mechanism accepted below streaming fit")
	}
}
