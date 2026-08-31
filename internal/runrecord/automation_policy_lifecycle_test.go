package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"testing"
)

func TestAutomationPolicyLifecycle(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	policy := testutil.ArtifactID(t, artifact.KindProfile, "candidate-policy")
	incumbent := testutil.ArtifactID(t, artifact.KindProfile, "incumbent-policy")
	base := AutomationPolicyLifecycle{Name: "agent-harness", Policy: policy, Incumbent: incumbent, Rollback: incumbent,
		EvaluationPlan: testutil.ArtifactID(t, artifact.KindProfile, "evaluation-plan"), EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "comparison"), Trajectories: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")}}
	states := []AutomationPolicyState{AutomationPolicyDeclared, AutomationPolicyExperimental, AutomationPolicyVerified, AutomationPolicyActive, AutomationPolicyContained, AutomationPolicyRolledBack}
	for _, state := range states {
		base.State = state
		base.Decision = testutil.ArtifactID(t, artifact.KindEvidence, string(state))
		base, err = PublishAutomationPolicyTransition(ctx, store, base)
		if err != nil {
			t.Fatalf("%s: %v", state, err)
		}
	}
	active, found, err := store.ResolveAlias(ctx, AutomationPolicyActiveAliasRoot+base.Name)
	if err != nil || !found || active != incumbent {
		t.Fatalf("rollback alias = %s %v %v", active, found, err)
	}
}
