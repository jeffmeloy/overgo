package automationcheck

import "time"

// ManifestMeasurements makes selector safety and efficiency observable.
type ManifestMeasurements struct {
	Defined        int   `json:"defined"`
	Selected       int   `json:"selected"`
	Excluded       int   `json:"excluded"`
	Uncertainty    int   `json:"uncertainty"`
	CacheEligible  int   `json:"cache_eligible"`
	CacheHits      int   `json:"cache_hits"`
	CacheMisses    int   `json:"cache_misses"`
	PlanningNS     int64 `json:"planning_ns"`
	AnalysisAgeNS  int64 `json:"analysis_age_ns"`
	FullPlanParity bool  `json:"full_plan_parity"`
}

// MeasureManifest records complete disposition and timing without treating
// exclusion rate as correctness evidence.
func MeasureManifest(defined, selected, excluded, uncertainty, eligible, hits int, planning, analysisAge time.Duration) ManifestMeasurements {
	return ManifestMeasurements{
		Defined: defined, Selected: selected, Excluded: excluded, Uncertainty: uncertainty,
		CacheEligible: eligible, CacheHits: hits, CacheMisses: eligible - hits,
		PlanningNS: planning.Nanoseconds(), AnalysisAgeNS: analysisAge.Nanoseconds(),
		FullPlanParity: selected+excluded == defined,
	}
}
