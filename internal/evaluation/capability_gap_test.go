package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestCapabilityGapTargetsFromEvalStore pins the target derivation the
// composition driver consumes: every cited document admits through the
// same evidence gate routing uses, the latest admitted measurement per
// recipe and metric stands, a latest below the store frontier is a
// frontier gap citing both sides, a latest below the recipe's own best
// earlier measurement is a regression, each recipe's largest shortfall is
// its weakest domain, targets order deterministically, and an unadmitted
// document refuses the whole derivation.
func TestCapabilityGapTargetsFromEvalStore(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	incumbentFirst, incumbentDefinition := publishSelectionEvidence(t, store, "gap-incumbent", 8192, 0.5)
	strongEvidence, strongDefinition := publishSelectionEvidence(t, store, "gap-strong", 16384, 0.9)
	weakEvidence, weakDefinition := publishSelectionEvidence(t, store, "gap-weak", 2048, 0.3)
	incumbentLatest := publishSelectionEvaluation(
		t, store, incumbentDefinition, "gap-incumbent", "regressed", 0.35,
	)
	corpus := []artifact.ID{incumbentFirst, strongEvidence, weakEvidence, incumbentLatest}

	targets, err := DeriveCapabilityGapTargets(ctx, store, corpus)
	if err != nil {
		t.Fatal(err)
	}
	byKind := map[CapabilityGapKind][]CapabilityGapTarget{}
	for _, target := range targets {
		byKind[target.Kind] = append(byKind[target.Kind], target)
	}
	if len(byKind[GapFrontier]) != 2 || len(byKind[GapRegression]) != 1 || len(byKind[GapWeakestDomain]) != 2 {
		t.Fatalf("target census = %+v", byKind)
	}
	regression := byKind[GapRegression][0]
	if regression.Recipe != incumbentDefinition.ID || regression.Measured != 0.35 ||
		regression.Reference != 0.5 || regression.Gap != regression.Reference-regression.Measured {
		t.Fatalf("regression target = %+v", regression)
	}
	cited := map[artifact.ID]bool{}
	for _, id := range regression.Evidence {
		cited[id] = true
	}
	if !cited[incumbentLatest] || !cited[incumbentFirst] {
		t.Fatalf("regression evidence = %v", regression.Evidence)
	}
	for _, frontier := range byKind[GapFrontier] {
		if frontier.Reference != 0.9 {
			t.Fatalf("frontier reference = %+v", frontier)
		}
		if frontier.Recipe == strongDefinition.ID {
			t.Fatalf("the frontier holder received a frontier gap: %+v", frontier)
		}
		if frontier.Recipe == incumbentDefinition.ID && frontier.Measured != 0.35 {
			t.Fatalf("frontier gap used a stale measurement: %+v", frontier)
		}
		if !cited[incumbentLatest] {
			t.Fatal("frontier citations lost")
		}
	}
	weakestByRecipe := map[artifact.ID]CapabilityGapTarget{}
	for _, weakestTarget := range byKind[GapWeakestDomain] {
		weakestByRecipe[weakestTarget.Recipe] = weakestTarget
	}
	if _, holderHasWeakness := weakestByRecipe[strongDefinition.ID]; holderHasWeakness {
		t.Fatal("the frontier holder received a weakest-domain target")
	}
	if weakestByRecipe[weakDefinition.ID].Gap <= weakestByRecipe[incumbentDefinition.ID].Gap {
		t.Fatalf("weakest-domain gaps out of order: %+v", weakestByRecipe)
	}

	again, err := DeriveCapabilityGapTargets(ctx, store, []artifact.ID{
		incumbentLatest, weakEvidence, strongEvidence, incumbentFirst,
	})
	if err != nil || len(again) != len(targets) {
		t.Fatalf("reordered corpus changed the targets: (%d vs %d, %v)", len(again), len(targets), err)
	}
	for index := range targets {
		if targets[index].Kind != again[index].Kind || targets[index].Recipe != again[index].Recipe ||
			targets[index].Gap != again[index].Gap {
			t.Fatalf("target order drifted at %d", index)
		}
	}

	forged := planID(t, artifact.KindEvidence, "gap-forged")
	if _, err := DeriveCapabilityGapTargets(ctx, store, []artifact.ID{incumbentFirst, forged}); err == nil {
		t.Fatal("unadmitted evidence derived targets")
	}
	if _, err := DeriveCapabilityGapTargets(ctx, store, nil); err == nil ||
		!strings.Contains(err.Error(), "requires admitted evaluation evidence") {
		t.Fatalf("empty corpus derived targets: %v", err)
	}
}
