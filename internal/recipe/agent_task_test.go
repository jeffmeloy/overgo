package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestAgentTaskContractIdentity(t *testing.T) {
	contract := AgentTaskContract{
		Agent: testutil.ArtifactID(t, artifact.KindRecipe, "agent"), Objective: "Tighten the agent harness.",
		Scope: []string{"internal/agentloop", "internal/agenttool"}, NonGoals: []string{"change model kernels"},
		AllowedEffects: []string{"workspace:write"},
		Acceptance:     []AcceptanceCriterion{{Name: "tests", Scope: "internal/agentloop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}},
		Verification:   []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"unresolved mutation target"},
	}
	first, err := NewAgentTaskContract(contract)
	if err != nil {
		t.Fatal(err)
	}
	contract.Scope[0], contract.Scope[1] = contract.Scope[1], contract.Scope[0]
	second, err := NewAgentTaskContract(contract)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || len(first.Lineage()) == 0 {
		t.Fatalf("task identities differ: %s %s", first.ID, second.ID)
	}
	contract.Acceptance = nil
	if _, err := NewAgentTaskContract(contract); err == nil {
		t.Fatal("contract without acceptance was admitted")
	}
}
