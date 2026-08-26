package automationcheck

import (
	"testing"
	"time"
)

func TestSelectionMeasurements(t *testing.T) {
	measurement := MeasureManifest(10, 7, 3, 2, 6, 4, time.Millisecond, 2*time.Millisecond)
	if !measurement.FullPlanParity || measurement.Uncertainty != 2 || measurement.CacheEligible != 6 ||
		measurement.CacheHits != 4 || measurement.CacheMisses != 2 ||
		measurement.PlanningNS <= 0 || measurement.AnalysisAgeNS <= measurement.PlanningNS {
		t.Fatalf("measurement = %+v", measurement)
	}
}
