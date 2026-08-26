package automationcheck

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codemanifest"
	"overgo/internal/runrecord"
)

func TestManifestAnalysisStrictAuthorityRoundTrip(t *testing.T) {
	plan := manifestAnalysisPlanFixture(t)
	delta := codemanifest.Delta{Base: plan.BaseManifest, Candidate: plan.CandidateManifest}
	impact := codemanifest.Impact{Base: plan.BaseManifest.String(), Candidate: plan.CandidateManifest.String()}
	measurements := MeasureManifest(1, 1, 0, 0, 1, 0, 0, 0)
	analysis, err := NewManifestAnalysis(delta, impact, plan, SelectionMetrics{}, measurements)
	if err != nil {
		t.Fatal(err)
	}
	content, err := analysis.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := ParseManifestAnalysis(content.Data)
	if err != nil || replayed.ID != analysis.ID {
		t.Fatalf("manifest analysis replay = (%s, %v)", replayed.ID, err)
	}
}

func TestManifestAnalysisRejectsMixedAuthority(t *testing.T) {
	plan := manifestAnalysisPlanFixture(t)
	other, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("other"))
	delta := codemanifest.Delta{Base: other, Candidate: plan.CandidateManifest}
	impact := codemanifest.Impact{Base: plan.BaseManifest.String(), Candidate: plan.CandidateManifest.String()}
	measurements := MeasureManifest(1, 1, 0, 0, 1, 0, 0, 0)
	if _, err := NewManifestAnalysis(delta, impact, plan, SelectionMetrics{}, measurements); err == nil {
		t.Fatal("mixed structural authority accepted")
	}
}

func manifestAnalysisPlanFixture(t *testing.T) ManifestPlan {
	t.Helper()
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	runner := func(context.Context, Invocation) (bool, string, error) { return true, "verified", nil }
	invocations, err := Plan([]Check{{Descriptor: Descriptor{Name: "verify", Phase: runrecord.PhaseTest, Always: true}, Run: runner}}, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64), Surface{Identity: "fixture"}, Impact{}, invocations)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}
