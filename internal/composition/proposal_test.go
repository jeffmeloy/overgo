package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestBridgeProposalBlocksWithoutVerifier pins the bridge-proposal contract:
// retrieval-derived candidates are advisory by construction -- every proposal
// is promotion-blocked with a stated blocker, requires a failable go-test
// verifier, and no constructor or decode path can represent an authorized
// state.
func TestBridgeProposalBlocksWithoutVerifier(t *testing.T) {
	target := testutil.ArtifactID(t, artifact.KindModel, "proposal-target")
	donor := testutil.ArtifactID(t, artifact.KindModel, "proposal-donor")
	ranker, err := TrainProposalRanker(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	candidates := []BridgeCandidate{{Donor: donor, Component: "blk.7.ffn_gate.weight", Distance: 0.125}}
	verifier := "go test ./internal/adaptiveparity -run '^TestComponentCompositionViability$' -count=1"
	blocker := "no recorded experiment evidence; the adapter tier is retired pending per-matrix modulation"

	proposal, err := NewBridgeProposal(target, candidates, ranker, verifier, blocker)
	if err != nil {
		t.Fatal(err)
	}
	if proposal.State != ProposalPromotionBlocked || proposal.Blocker != blocker {
		t.Fatalf("proposal state = %+v, want promotion-blocked with blocker", proposal)
	}
	replay, err := NewBridgeProposal(target, candidates, ranker, verifier, blocker)
	if err != nil || replay.ID != proposal.ID {
		t.Fatalf("replay = (%v, %v), want identical identity %v", replay.ID, err, proposal.ID)
	}

	if _, err := NewBridgeProposal(target, candidates, ranker, "", blocker); err == nil {
		t.Fatal("proposal without a verifier accepted")
	}
	if _, err := NewBridgeProposal(target, candidates, ranker, "echo ok", blocker); err == nil {
		t.Fatal("non-go-test verifier accepted")
	}
	if _, err := NewBridgeProposal(target, candidates, ranker, "go test ./internal/composition -count=1", blocker); err == nil {
		t.Fatal("verifier without a -run acceptance target accepted")
	}
	if _, err := NewBridgeProposal(target, candidates, ranker, verifier, "  "); err == nil {
		t.Fatal("proposal without a blocker reason accepted")
	}
	if _, err := NewBridgeProposal(target, nil, ranker, verifier, blocker); err == nil {
		t.Fatal("candidate-free proposal accepted")
	}

	content, err := proposal.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseBridgeProposal(content.Data)
	if err != nil || parsed.ID != proposal.ID || parsed.State != ProposalPromotionBlocked {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
	authorized := strings.Replace(string(content.Data), ProposalPromotionBlocked, "authorized", 1)
	if _, err := ParseBridgeProposal([]byte(authorized)); err == nil {
		t.Fatal("decode accepted an authorized state; the blocked state must be unrepresentable")
	}

	batch, err := proposal.Batch("bridge-proposal/" + proposal.ID.String())
	if err != nil || len(batch.Contents) != 1 || len(batch.Lineage) != 3 {
		t.Fatalf("batch = (%+v, %v), want content plus target and donor lineage", batch, err)
	}
}
