package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestProposalRankingImprovesOnRefusalHistory(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	target := testutil.ArtifactID(t, artifact.KindModel, "ranking-target")
	bad := BridgeCandidate{
		Donor:     testutil.ArtifactID(t, artifact.KindModel, "ranking-refused"),
		Component: "model.layers.8.mlp.down_proj.weight", Distance: 0.1,
	}
	good := BridgeCandidate{
		Donor:     testutil.ArtifactID(t, artifact.KindModel, "ranking-observed"),
		Component: "model.layers.11.mlp.gate_proj.weight", Distance: 0.2,
	}
	cold := testcheck.Must(TrainProposalRanker(nil, nil, nil))(t)
	verifier := "go test ./internal/composition -run '^TestProposalRankingImprovesOnRefusalHistory$' -count=1"
	proposal := func(candidate BridgeCandidate, blocker string) BridgeProposal {
		value := testcheck.Must(NewBridgeProposal(target, []BridgeCandidate{candidate}, cold, verifier, blocker))(t)
		return value
	}
	refusedProposal := proposal(bad, "measured operator mismatch")
	observedProposal := proposal(good, "trial remains promotion-blocked")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "ranking measurement")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "ranking decider")
	decider := recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: derivation}
	refused := testcheck.Must(recipe.NewDecision(refusedProposal.ID, recipe.DecisionRefused, recipe.EvidenceVerified,
		"operator contract mismatch", decider, []artifact.ID{evidence}))(t)
	observed := testcheck.Must(recipe.NewDecision(observedProposal.ID, recipe.DecisionObserved, recipe.EvidenceVerified,
		"bounded trial completed", decider, []artifact.ID{evidence}))(t)
	coldContent := testcheck.Must(cold.Content())(t)
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

	ranker := testcheck.Must(TrainProposalRankerFromStore(ctx, store))(t)
	ranked := ranker.Rank([]BridgeCandidate{bad, good})
	if len(ranked) != 2 || ranked[0].Donor != good.Donor || ranked[1].Donor != bad.Donor {
		t.Fatalf("refusal-ledger ranking = %+v", ranked)
	}
	if len(ranker.Rows) != 2 || len(ranker.Decisions) != 2 {
		t.Fatalf("ranker evidence = %+v", ranker)
	}
	accepted := testcheck.Must(recipe.NewDecision(observedProposal.ID, recipe.DecisionAccepted, recipe.EvidenceParity,
		"promotion result", decider, []artifact.ID{evidence}))(t)
	acceptedBatch := testcheck.Must(accepted.Batch("proposal-ranking-promotion"))(t)
	if _, err := store.Commit(ctx, acceptedBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := TrainProposalRanker(ctx, store, []artifact.ID{accepted.ID}); err == nil {
		t.Fatal("promotion outcome entered proposer training")
	}
}
