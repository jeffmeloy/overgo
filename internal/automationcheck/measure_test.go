package automationcheck

import (
	"testing"
	"time"
)

func TestSelectionMeasurements(t *testing.T) {
	measurement := MeasureManifest(10, 7, 3, 2, 4, 6, time.Millisecond, 2*time.Millisecond)
	if !measurement.FullPlanParity || measurement.Uncertainty != 2 || measurement.CacheHits != 4 ||
		measurement.PlanningNS <= 0 || measurement.GateElapsedNS <= measurement.PlanningNS {
		t.Fatalf("measurement = %+v", measurement)
	}
}
