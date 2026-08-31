package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func recordRecipeRoutingDecision(
	t *testing.T, store artifact.Repository, name string,
	threshold float64, candidates []RoutingCandidate,
) RecipeRoutingDecision {
	t.Helper()
	ctx := t.Context()
	thresholdEvidence := testutil.ArtifactID(t, artifact.KindEvidence, name+"-threshold")
	testutil.PublishArtifact(t, store, thresholdEvidence)
	for _, candidate := range candidates {
		testutil.PublishArtifact(t, store, candidate.Recipe)
		testutil.PublishArtifact(t, store, candidate.Model)
		for _, id := range candidate.Evidence {
			testutil.PublishArtifact(t, store, id)
		}
	}
	decision, err := DeriveRecipeRoutingDecision(RoutingSignal{
		Task: recipe.TaskInference, Capability: "replay-capability",
		QualityThreshold: threshold, ThresholdEvidence: thresholdEvidence,
	}, candidates)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := decision.Batch("routing/replay-fixture/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return decision
}

// TestRoutingCounterfactualReplay pins the counterfactual contract: a
// candidate policy is measured against the recorded decision corpus on
// identical inputs, the published report carries exact selection-change
// counts and quality and resource deltas with complete coverage, replaying
// the incumbent policy measures zero delta, and a corpus that is empty,
// names an unrecorded decision, or cites an unregistered policy refuses
// instead of shrinking the measurement.
func TestRoutingCounterfactualReplay(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	cheap := routingCandidateFixture(t, "first-cheap", 1024)
	costly := RoutingCandidate{
		Recipe:        testutil.ArtifactID(t, artifact.KindRecipe, "first-costly"),
		Model:         testutil.ArtifactID(t, artifact.KindModel, "routing-model"),
		Evidence:      []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "first-costly-evaluation")},
		Quality:       0.9,
		ResourceBytes: 4096,
	}
	first := recordRecipeRoutingDecision(t, store, "first", 0.6, []RoutingCandidate{cheap, costly})
	second := recordRecipeRoutingDecision(t, store, "second", 0.6, []RoutingCandidate{
		routingCandidateFixture(t, "second-only", 2048),
	})
	corpus := []artifact.ID{first.ID, second.ID}

	report, err := ReplayRoutingDecisions(ctx, store, RoutingDerivationBestQualityEligible, corpus)
	if err != nil {
		t.Fatal(err)
	}
	if report.Changed != 1 || report.Coverage != 2 ||
		report.QualityDelta != costly.Quality-cheap.Quality ||
		report.ResourceDelta != int64(costly.ResourceBytes)-int64(cheap.ResourceBytes) {
		t.Fatalf("counterfactual report = %+v", report)
	}
	content, found, err := artifact.ReadContent(ctx, store, report.ID)
	if err != nil || !found {
		t.Fatalf("counterfactual was not published: (%v, %v)", found, err)
	}
	published, err := routingCounterfactualCodec.Parse(content.Data)
	if err != nil || published.ID != report.ID || published.Changed != 1 {
		t.Fatalf("published counterfactual = (%+v, %v)", published, err)
	}

	incumbent, err := ReplayRoutingDecisions(ctx, store, RoutingDerivationCheapestEligible, corpus)
	if err != nil || incumbent.Changed != 0 || incumbent.QualityDelta != 0 || incumbent.ResourceDelta != 0 {
		t.Fatalf("incumbent replay = (%+v, %v)", incumbent, err)
	}

	if _, err := ReplayRoutingDecisions(
		ctx, store, RoutingDerivationBestQualityEligible, nil,
	); err == nil || !strings.Contains(err.Error(), "recorded decision corpus") {
		t.Fatalf("empty corpus measured: %v", err)
	}
	if _, err := ReplayRoutingDecisions(
		ctx, store, "hand-tuned/v0", corpus,
	); err == nil || !strings.Contains(err.Error(), "unregistered routing derivation") {
		t.Fatalf("unregistered policy replayed: %v", err)
	}
	unrecorded := testutil.ArtifactID(t, artifact.KindEvidence, "never-recorded")
	if _, err := ReplayRoutingDecisions(
		ctx, store, RoutingDerivationBestQualityEligible, []artifact.ID{first.ID, unrecorded},
	); err == nil || !strings.Contains(err.Error(), "is absent") {
		t.Fatalf("unrecorded corpus entry measured: %v", err)
	}
}
