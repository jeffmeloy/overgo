package agentworkflow

import (
	"context"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/runrecord"
)

func TestAgentManifestContext(t *testing.T) {
	impact, plan := agentManifestFixture(t)
	prior, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("prior"))
	context, err := NewManifestContext(impact, plan, []artifact.ID{prior})
	if err != nil {
		t.Fatal(err)
	}
	if context.Plan != plan.ID || len(context.AffectedSymbols) != 1 || len(context.RiskBoundaries) != 1 ||
		len(context.RequiredChecks) != 1 || !slices.Equal(context.PriorEvidence, []artifact.ID{prior}) {
		t.Fatalf("agent manifest context = %+v", context)
	}
}

func TestAgentCannotAuthorizeExclusion(t *testing.T) {
	impact, plan := agentManifestFixture(t)
	context, err := NewManifestContext(impact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(context.ProvenExclusions) != len(plan.Exclusions) {
		t.Fatal("agent context changed selector exclusions")
	}
	plan.Exclusions = append(plan.Exclusions, automationcheck.Exclusion{Check: "verify", Reason: "agent says safe"})
	if _, err := NewManifestContext(impact, plan, nil); err == nil {
		t.Fatal("agent-authored exclusion survived manifest-plan identity validation")
	}
}

func agentManifestFixture(t *testing.T) (codemanifest.Impact, automationcheck.ManifestPlan) {
	t.Helper()
	runner := func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil }
	checks := []automationcheck.Check{{Descriptor: automationcheck.Descriptor{
		Name: "verify", Phase: runrecord.PhaseTest, Always: true,
	}, Run: runner}}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	plan, err := automationcheck.BindManifestPlan(
		base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64),
		automationcheck.Surface{Identity: "surface"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	symbol := codemanifest.SymbolID{Package: "internal/model", Context: "windows/amd64", Name: "Run", Kind: codemanifest.SymbolFunction}
	impact := codemanifest.Impact{
		Base: base.String(), Candidate: candidate.String(), Reachable: []codemanifest.SymbolID{symbol}, Packages: []string{"internal/model"},
		Uncertainty: []codemanifest.Uncertainty{{Kind: codemanifest.UncertaintyReflection, Symbol: &symbol, Reason: "dynamic dispatch"}},
	}
	return impact, plan
}
