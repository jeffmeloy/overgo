package workflowruntime

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestRSICausalChainClosure pins the delegation side of the chain: a
// compiled delegation derives the child's causal context from the
// delegating execution, copying the root verbatim and naming the
// delegator as the causal subject, while granting nothing.
func TestRSICausalChainClosure(t *testing.T) {
	proposal := testutil.ArtifactID(t, artifact.KindEvidence, "delegation-proposal")
	delegator := testutil.ArtifactID(t, artifact.KindEvidence, "delegating-execution")
	root, err := runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, proposal)
	if err != nil {
		t.Fatal(err)
	}
	child, err := CompiledDelegation{}.DeriveCausal(root, delegator)
	if err != nil {
		t.Fatal(err)
	}
	if child.Trigger != runrecord.TriggerDelegation || child.Root != proposal || child.DelegatedFrom != delegator {
		t.Fatalf("delegated causal context = %+v", child)
	}
	// A second hop still answers to the first root: delegation depth
	// never mints a new causal principal.
	grandchild, err := CompiledDelegation{}.DeriveCausal(child, testutil.ArtifactID(t, artifact.KindEvidence, "child-execution"))
	if err != nil {
		t.Fatal(err)
	}
	if grandchild.Root != proposal {
		t.Fatalf("delegation re-minted the causal principal: %+v", grandchild)
	}
}
