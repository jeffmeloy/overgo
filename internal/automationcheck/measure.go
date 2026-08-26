package automationcheck

import "time"

// ManifestMeasurements makes selector safety and efficiency observable.
type ManifestMeasurements struct {
	Defined        int   `json:"defined"`
	Selected       int   `json:"selected"`
	Excluded       int   `json:"excluded"`
	Uncertainty    int   `json:"uncertainty"`
	CacheHits      int   `json:"cache_hits"`
	CacheMisses    int   `json:"cache_misses"`
	PlanningNS     int64 `json:"planning_ns"`
	GateElapsedNS  int64 `json:"gate_elapsed_ns"`
	FullPlanParity bool  `json:"full_plan_parity"`
}

// MeasureManifest records complete disposition and timing without treating
// exclusion rate as correctness evidence.
func MeasureManifest(defined, selected, excluded, uncertainty, hits, misses int, planning, elapsed time.Duration) ManifestMeasurements {
	return ManifestMeasurements{
		Defined: defined, Selected: selected, Excluded: excluded, Uncertainty: uncertainty,
		CacheHits: hits, CacheMisses: misses, PlanningNS: planning.Nanoseconds(), GateElapsedNS: elapsed.Nanoseconds(),
		FullPlanParity: selected+excluded == defined,
	}
}
