package controlleraction

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
)

// TestControllerActionsAreAllowlistedTransformations pins the action-language
// contract: every allowlisted kind compiles deterministically through the
// shared authorities into content-addressed artifacts; unknown kinds and
// kind/payload mismatches are unrepresentable; and the chain emission is
// genuinely executable through the generic workflow runtime's catalog.
func TestControllerActionsAreAllowlistedTransformations(t *testing.T) {
	scorer := testutil.ArtifactID(t, artifact.KindModel, "action-scorer")
	drafter := testutil.ArtifactID(t, artifact.KindModel, "action-drafter")
	chain := Action{Version: ActionVersion, Kind: KindChainRecipes, Chain: &ChainAction{Scorer: scorer, Drafter: drafter}}
	first, err := Compile(chain)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(chain)
	if err != nil || len(first) != 2 || len(second) != 2 ||
		first[0].Descriptor.ID != second[0].Descriptor.ID || first[1].Descriptor.ID != second[1].Descriptor.ID {
		t.Fatalf("chain compile not deterministic: (%v, %v)", second, err)
	}
	definition, err := recipe.ParseDefinition(first[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recipe.CompileProgram(definition, workflowrecipe.Catalog()); err != nil {
		t.Fatalf("emitted chain recipe is not executable through the catalog: %v", err)
	}

	target := testutil.ArtifactID(t, artifact.KindModel, "action-target")
	donor := testutil.ArtifactID(t, artifact.KindModel, "action-donor")
	proposal := Action{Version: ActionVersion, Kind: KindBridgeProposal, Proposal: &ProposalAction{
		Target:           target,
		Candidates:       []composition.BridgeCandidate{{Donor: donor, Component: "blk.3.ffn_gate.weight", Distance: 0.5}},
		RequiredVerifier: "go test ./internal/adaptiveparity -run '^TestComponentCompositionViability$' -count=1",
		Blocker:          "no recorded experiment evidence",
	}}
	contents, err := Compile(proposal)
	if err != nil || len(contents) != 1 {
		t.Fatalf("proposal compile = (%d, %v)", len(contents), err)
	}
	parsed, err := composition.ParseBridgeProposal(contents[0].Data)
	if err != nil || parsed.State != composition.ProposalPromotionBlocked {
		t.Fatalf("compiled proposal = (%+v, %v), want promotion-blocked", parsed, err)
	}

	model := testutil.ArtifactID(t, artifact.KindModel, "action-model")
	decomposition := Action{Version: ActionVersion, Kind: KindComponentDecomposition, Decomposition: &DecompositionAction{
		Model:   model,
		Family:  "qwen2",
		Tensors: []modelartifact.TensorFact{{Name: "blk.0.attn_q.weight", Shape: []uint64{8, 8}, Storage: "f32"}},
	}}
	contents, err = Compile(decomposition)
	if err != nil || len(contents) != 1 {
		t.Fatalf("decomposition compile = (%d, %v)", len(contents), err)
	}

	if _, err := Compile(Action{Version: ActionVersion, Kind: "run-shell", Chain: chain.Chain}); err == nil {
		t.Fatal("kind outside the allowlist compiled")
	}
	if _, err := Compile(Action{Version: ActionVersion, Kind: KindChainRecipes, Proposal: proposal.Proposal}); err == nil {
		t.Fatal("kind/payload mismatch compiled")
	}
	if _, err := Compile(Action{Version: ActionVersion, Kind: KindChainRecipes, Chain: chain.Chain, Proposal: proposal.Proposal}); err == nil {
		t.Fatal("multiple payloads compiled")
	}
	if _, err := ParseAction([]byte(`{"version":1,"kind":"chain-recipes","chain":{"scorer":"` +
		scorer.String() + `","drafter":"` + drafter.String() + `"},"shell":"rm -rf /"}`)); err == nil {
		t.Fatal("action document with a code-carrying field parsed; the language must refuse unknown fields")
	}

	serialized, err := json.Marshal(chain)
	if err != nil {
		t.Fatal(err)
	}
	roundtrip, err := ParseAction(serialized)
	if err != nil || roundtrip.Kind != KindChainRecipes || roundtrip.Chain == nil {
		t.Fatalf("roundtrip = (%+v, %v)", roundtrip, err)
	}
}
