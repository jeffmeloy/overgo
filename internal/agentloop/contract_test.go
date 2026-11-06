package agentloop

import (
	"encoding/json"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestAgentContractEnforcement(t *testing.T) {
	contract := testcheck.Must(recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: testutil.ArtifactID(t, artifact.KindRecipe, "agent"), Objective: "Change one package.",
		Scope: []string{"internal/agentloop"}, AllowedEffects: []string{"workspace:mutation"},
		Acceptance:   []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/agentloop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "tests")}},
		Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"opaque mutation"},
	}))(t)
	obligation := testcheck.Must(runrecord.NewAgentObligation(runrecord.AgentObligation{
		Task: contract.ID, Name: "tests", Scope: "internal/agentloop", Sources: []artifact.ID{contract.ID},
	}))(t)
	state := testcheck.Must(NewContractState(contract, []runrecord.AgentObligation{obligation}))(t)
	if len(state.epochs) != 0 {
		t.Fatalf("a fresh contract state holds the epochs %v, want none", state.epochs)
	}
	manual := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "file.write", Description: "Write one file.", Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{{Name: "path", Kind: agenttool.FieldString, Required: true}},
		Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: "path"}}},
		Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	}))(t)
	effect := testcheck.Must(agenttool.DeriveInvocationEffect(manual, json.RawMessage(`{"path":"internal/agentloop/coordinator.go"}`), nil))(t)
	testcheck.NoError(t, state.Admit(effect))
	state.ObserveMutation(effect)
	if len(state.epochs) != 1 || state.epochs[obligation.Scope] != 1 {
		t.Fatalf("a mutation inside the obligation's scope advanced the epochs %v, want its scope's advanced once, which leaves its evidence stale", state.epochs)
	}
}
