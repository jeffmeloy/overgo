package storepolicy

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// TestTelemetryKeepsAHorizon holds gate telemetry to its own class: both
// schemas that explain a gate's cost are telemetry, none is a derived cache,
// since git cannot recompute them, none of the gate's decision evidence is
// telemetry, and a release keeps a horizon of them rather than all or none.
func TestTelemetryKeepsAHorizon(t *testing.T) {
	t.Parallel()
	for _, schema := range []string{runrecord.SelectionCauseSchema, runrecord.SuiteCostSchema} {
		if !Telemetry(artifact.Descriptor{Schema: schema}) {
			t.Errorf("%s is not classed as telemetry", schema)
		}
		if slices.Contains(DerivedCacheSchemas, schema) {
			t.Errorf("%s is telemetry and is classed as a derived cache", schema)
		}
	}
	for _, schema := range []string{runrecord.AttemptSchema, runrecord.GateSchema, runrecord.GateLifecycleSchema} {
		if Telemetry(artifact.Descriptor{Schema: schema}) {
			t.Errorf("gate decision evidence %s is classed as telemetry", schema)
		}
	}
	if TelemetryHorizon <= 0 {
		t.Fatalf("telemetry horizon = %d keeps nothing", TelemetryHorizon)
	}
}
