package agentloop

import (
	"encoding/json"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestAgentContractEnforcement(t *testing.T) {
	contract, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: testutil.ArtifactID(t, artifact.KindRecipe, "agent"), Objective: "Change one package.",
		Scope: []string{"internal/agentloop"}, AllowedEffects: []string{"workspace:mutation"},
		Acceptance:   []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/agentloop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "tests")}},
		Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"opaque mutation"},
	})
	if err != nil {
		t.Fatal(err)
	}
	obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
		Task: contract.ID, Name: "tests", Scope: "internal/agentloop", Sources: []artifact.ID{contract.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewContractState(contract, []runrecord.AgentObligation{obligation})
	if err != nil {
		t.Fatal(err)
	}
	resolution, err := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{
		Obligation: obligation.ID, Scope: obligation.Scope, Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "pass")},
	})
	if err != nil {
		t.Fatal(err)
	}
	state.AddResolution(resolution)
	state.AddResolution(resolution)
	foreign, err := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{
		Obligation: testutil.ArtifactID(t, artifact.KindEvidence, "foreign obligation"), Scope: obligation.Scope,
		Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "foreign pass")},
	})
	if err != nil {
		t.Fatal(err)
	}
	state.AddResolution(foreign)
	if len(state.resolutions) != 1 || len(state.epochs) != 0 {
		t.Fatalf("contract state is not obligation-bounded: resolutions=%d epochs=%d", len(state.resolutions), len(state.epochs))
	}
	if err := state.RequireComplete(); err != nil {
		t.Fatal(err)
	}
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "file.write", Description: "Write one file.", Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{{Name: "path", Kind: agenttool.FieldString, Required: true}},
		Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: "path"}}},
		Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	effect, err := agenttool.DeriveInvocationEffect(manual, json.RawMessage(`{"path":"internal/agentloop/coordinator.go"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Admit(effect); err != nil {
		t.Fatal(err)
	}
	state.ObserveMutation(effect)
	if len(state.Outstanding()) != 1 {
		t.Fatal("later relevant mutation did not invalidate evidence")
	}
}
