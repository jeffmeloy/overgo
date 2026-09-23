package storepolicy

import (
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// TelemetrySchemas explain how a gate spent its time -- which packages it
// selected and why, what each suite cost -- and prove nothing: no gate
// decision reads them, only the history and the optimization review. Git
// cannot recompute them, so they are not derived caches; a release keeps the
// newest TelemetryHorizon records of each and may drop the rest.
var TelemetrySchemas = []string{runrecord.SelectionCauseSchema, runrecord.SuiteCostSchema}

// TelemetryHorizon is how many of each telemetry schema's newest records a
// release keeps: about two days of landings at the measured 2026-09-22 rate
// of 70 landings between store swaps, enough for the history and the suite
// regression review to compare a gate with the ones before it.
const TelemetryHorizon = 70

// Telemetry reports a descriptor whose bytes are gate telemetry.
func Telemetry(descriptor artifact.Descriptor) bool {
	return slices.Contains(TelemetrySchemas, descriptor.Schema)
}
