package main

import (
	"math"
	"strings"
	"testing"
)

func TestSweepBaselineComparisonRejectsMaterialRegression(t *testing.T) {
	baseline := sweepBaseline{
		RegressionScale: 100, MaximumWallIncrease: 50, MaximumPeakDeviceIncrease: 5,
	}
	target := sweepTarget{PromptTokens: 10, GeneratedTokens: 4, WallNS: 1_000, PeakDeviceBytes: 2_000}
	for _, test := range []struct {
		name    string
		current report
		want    string
	}{
		{"within ceilings", report{PromptTokens: 10, GeneratedTokens: 4, WallNS: 1_500, PeakDeviceBytes: 2_100}, ""},
		{"wall regression", report{PromptTokens: 10, GeneratedTokens: 4, WallNS: 1_501, PeakDeviceBytes: 2_100}, "wall_ns"},
		{"memory regression", report{PromptTokens: 10, GeneratedTokens: 4, WallNS: 1_500, PeakDeviceBytes: 2_101}, "peak_device_bytes"},
		{"token drift", report{PromptTokens: 9, GeneratedTokens: 4, WallNS: 1_000, PeakDeviceBytes: 2_000}, "token counts changed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := compareSweep(target, test.current, baseline)
			if test.want == "" && err != nil {
				t.Fatal(err)
			}
			if test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("error=%v, want %q", err, test.want)
			}
		})
	}
}

func TestSweepBaselineValidationAndOverflow(t *testing.T) {
	target := sweepTarget{
		Name: "fixture", ModelPath: "model.gguf", PromptTokens: 10, GeneratedTokens: 4,
		WallNS: 1_000, PeakDeviceBytes: 2_000,
	}
	baseline := sweepBaseline{
		Schema: sweepBaselineSchema, SourceCommit: "fixture-commit", Fixture: "fixture.json", RegressionScale: 100,
		MaximumWallIncrease: 50, MaximumPeakDeviceIncrease: 5, Models: []sweepTarget{target},
	}
	if err := validateSweepBaseline(baseline, "fixture.json"); err != nil {
		t.Fatal(err)
	}
	baseline.Models = append(baseline.Models, target)
	if err := validateSweepBaseline(baseline, "fixture.json"); err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("duplicate error=%v", err)
	}
	if _, ok := regressionCeiling(math.MaxUint64, math.MaxUint64, 1); ok {
		t.Fatal("overflowing regression ceiling accepted")
	}
}
