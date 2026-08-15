package main

import (
	"math"
	"testing"

	"overgo/internal/runrecord"
)

func TestAdvisoryCalibrationContract(t *testing.T) {
	for _, test := range []struct {
		name         string
		observations int
		budget       float64
		required     int
		available    int
		sufficient   bool
	}{
		{name: "five percent needs nineteen held-out scores", observations: 80, budget: .05, required: 19, available: 19, sufficient: true},
		{name: "latest block is not calibration evidence", observations: 79, budget: .05, required: 19, available: 18},
		{name: "large budget still needs a useful empirical series", observations: 12, budget: .75, required: 2, available: 2, sufficient: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			plan := planCalibration(test.observations, test.budget)
			if plan.Window != runrecord.MinAdvisoryWindow || plan.RequiredScores != test.required ||
				plan.AvailableScores != test.available || plan.Sufficient() != test.sufficient {
				t.Fatalf("planCalibration() = %+v sufficient=%v", plan, plan.Sufficient())
			}
		})
	}

	observations := make([]runrecord.Observation, 9)
	if got := disjointSurprises(observations, "missing", runrecord.MinAdvisoryWindow); got != nil {
		t.Fatalf("invalid series returned %v, want nil", got)
	}
	if got := empiricalQuantile([]float64{4, 1, 3, 2}, .75); math.Abs(got-3) > 0 {
		t.Fatalf("empiricalQuantile() = %g, want 3", got)
	}
}

func TestNonOverlappingConfirmation(t *testing.T) {
	latest := runrecord.Advisory{WindowStart: 20, WindowEnd: 22, LatestSequence: 23}
	for _, test := range []struct {
		name  string
		prior runrecord.Advisory
		want  bool
	}{
		{name: "disjoint", prior: runrecord.Advisory{WindowStart: 16, WindowEnd: 18, LatestSequence: 19}, want: true},
		{name: "shared boundary", prior: runrecord.Advisory{WindowStart: 17, WindowEnd: 19, LatestSequence: 20}},
		{name: "overlapping", prior: runrecord.Advisory{WindowStart: 18, WindowEnd: 20, LatestSequence: 21}},
		{name: "same latest", prior: latest},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := nonOverlappingConfirmation(test.prior, latest); got != test.want {
				t.Fatalf("nonOverlappingConfirmation() = %v, want %v", got, test.want)
			}
		})
	}
}
