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

// TestStoredAnalysisOmitsImpactLists holds the stored analysis to the facts its
// readers consume: the reachable symbols and their uncertainty stay out of its
// bytes however many there are, so its size does not grow with them, while
// the seeds, the packages and the plan its completion reads remain.
func TestStoredAnalysisOmitsImpactLists(t *testing.T) {
	plan := manifestAnalysisPlanFixture(t)
	delta := codemanifest.Delta{Base: plan.BaseManifest, Candidate: plan.CandidateManifest}
	measurements := MeasureManifest(1, 1, 0, 0, 1, 0, 0, 0)
	stored := func(reachable int) []byte {
		t.Helper()
		impact := codemanifest.Impact{
			Base: plan.BaseManifest.String(), Candidate: plan.CandidateManifest.String(),
			Seeds:    []codemanifest.SymbolID{{Package: "overgo/seed", Name: "Changed", Kind: codemanifest.SymbolFunction}},
			Packages: []string{"overgo/seed"},
		}
		for symbol := range reachable {
			reached := codemanifest.SymbolID{Package: "overgo/reached", Name: "Symbol" + strings.Repeat("x", symbol), Kind: codemanifest.SymbolFunction}
			impact.Reachable = append(impact.Reachable, reached)
			impact.Uncertainty = append(impact.Uncertainty, codemanifest.Uncertainty{Kind: codemanifest.UncertaintyReflection, Symbol: &reached, Reason: "reached"})
		}
		analysis, err := NewManifestAnalysis(delta, impact, plan, SelectionMetrics{}, measurements)
		if err != nil {
			t.Fatal(err)
		}
		content, err := analysis.Content()
		if err != nil {
			t.Fatal(err)
		}
		replayed, err := ParseManifestAnalysis(content.Data)
		if err != nil || replayed.Plan.ID != plan.ID || len(replayed.Impact.Seeds) != 1 || len(replayed.Impact.Packages) != 1 {
			t.Fatalf("stored analysis lost what its readers consume: %+v %v", replayed.Impact, err)
		}
		if len(replayed.Impact.Reachable)+len(replayed.Impact.Uncertainty) != 0 {
			t.Fatalf("stored analysis kept %d reachable symbols and %d uncertainties", len(replayed.Impact.Reachable), len(replayed.Impact.Uncertainty))
		}
		return content.Data
	}
	if few, many := stored(1), stored(200); len(few) != len(many) {
		t.Fatalf("stored analysis grew from %d to %d bytes with the reachable symbols", len(few), len(many))
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
