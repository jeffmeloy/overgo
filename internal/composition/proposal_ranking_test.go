package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestProposalRankingImprovesOnRefusalHistory(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	target := testutil.ArtifactID(t, artifact.KindModel, "ranking-target")
	bad := BridgeCandidate{
		Donor:     testutil.ArtifactID(t, artifact.KindModel, "ranking-refused"),
		Component: "model.layers.8.mlp.down_proj.weight", Distance: 0.1,
	}
	good := BridgeCandidate{
		Donor:     testutil.ArtifactID(t, artifact.KindModel, "ranking-observed"),
		Component: "model.layers.11.mlp.gate_proj.weight", Distance: 0.2,
	}
	cold, err := TrainProposalRanker(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	verifier := "go test ./internal/composition -run '^TestProposalRankingImprovesOnRefusalHistory$' -count=1"
	proposal := func(candidate BridgeCandidate, blocker string) BridgeProposal {
		value, err := NewBridgeProposal(target, []BridgeCandidate{candidate}, cold, verifier, blocker)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	refusedProposal := proposal(bad, "measured operator mismatch")
	observedProposal := proposal(good, "trial remains promotion-blocked")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "ranking measurement")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "ranking decider")
	decider := recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: derivation}
	refused, err := recipe.NewDecision(refusedProposal.ID, recipe.DecisionRefused, recipe.EvidenceVerified,
		"operator contract mismatch", decider, []artifact.ID{evidence})
	if err != nil {
		t.Fatal(err)
	}
	observed, err := recipe.NewDecision(observedProposal.ID, recipe.DecisionObserved, recipe.EvidenceVerified,
		"bounded trial completed", decider, []artifact.ID{evidence})
	if err != nil {
		t.Fatal(err)
	}
	coldContent, err := cold.Content()
	if err != nil {
		t.Fatal(err)
	}
	refusedContent, _ := refusedProposal.Content()
	observedContent, _ := observedProposal.Content()
	refusedDecision, _ := refused.Content()
	observedDecision, _ := observed.Content()
	lineage := append(cold.Lineage(), refusedProposal.Lineage()...)
	lineage = append(lineage, observedProposal.Lineage()...)
	lineage = append(lineage, refused.Lineage()...)
	lineage = append(lineage, observed.Lineage()...)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "proposal-ranking-history",
		Artifacts: []artifact.Descriptor{
			{ID: target}, {ID: bad.Donor}, {ID: good.Donor}, {ID: evidence}, {ID: derivation},
		},
		Contents: []artifact.Content{coldContent, refusedContent, observedContent, refusedDecision, observedDecision},
		Lineage:  lineage,
	}); err != nil {
		t.Fatal(err)
	}

	ranker, err := TrainProposalRankerFromStore(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	ranked := ranker.Rank([]BridgeCandidate{bad, good})
	if len(ranked) != 2 || ranked[0].Donor != good.Donor || ranked[1].Donor != bad.Donor {
		t.Fatalf("refusal-ledger ranking = %+v", ranked)
	}
	if len(ranker.Rows) != 2 || len(ranker.Decisions) != 2 {
		t.Fatalf("ranker evidence = %+v", ranker)
	}
	accepted, err := recipe.NewDecision(observedProposal.ID, recipe.DecisionAccepted, recipe.EvidenceParity,
		"promotion result", decider, []artifact.ID{evidence})
	if err != nil {
		t.Fatal(err)
	}
	acceptedBatch, err := accepted.Batch("proposal-ranking-promotion")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, acceptedBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := TrainProposalRanker(ctx, store, []artifact.ID{accepted.ID}); err == nil {
		t.Fatal("promotion outcome entered proposer training")
	}
}
