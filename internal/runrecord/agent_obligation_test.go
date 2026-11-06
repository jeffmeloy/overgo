package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"testing"
)

func TestAgentObligationEvidence(t *testing.T) {
	source := testutil.ArtifactID(t, artifact.KindEvidence, "receipt")
	obligation := testcheck.Must(NewAgentObligation(AgentObligation{
		Task: testutil.ArtifactID(t, artifact.KindRecipe, "task"), Name: "tests", Scope: "internal/agentloop",
		MutationEpoch: 3, Sources: []artifact.ID{source},
	}))(t)
	resolution := testcheck.Must(NewAgentObligationResolution(AgentObligationResolution{
		Obligation: obligation.ID, Scope: obligation.Scope, MutationEpoch: obligation.MutationEpoch, Evidence: []artifact.ID{source},
	}))(t)
	if !AgentObligationSatisfied(obligation, []AgentObligationResolution{resolution}) {
		t.Fatal("exact resolution remained open")
	}
	resolution.MutationEpoch++
	if AgentObligationSatisfied(obligation, []AgentObligationResolution{resolution}) {
		t.Fatal("stale resolution satisfied changed scope")
	}
	if _, err := NewAgentObligationResolution(AgentObligationResolution{Obligation: obligation.ID, Scope: obligation.Scope}); err == nil {
		t.Fatal("evidence-free resolution was admitted")
	}
}
