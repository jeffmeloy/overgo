package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"testing"
)

func TestAgentTrajectoryIdentity(t *testing.T) {
	base, err := NewInteractionTrace(Interaction{Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe"), Model: testutil.ArtifactID(t, artifact.KindModel, "model")},
		testutil.ArtifactID(t, artifact.KindEvidence, "request"), []InteractionMessage{{Role: "user", Content: "task"}, {Role: "assistant", Content: "result"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	base.Events[1].Branch = "candidate/a"
	base.TaskContract = testutil.ArtifactID(t, artifact.KindRecipe, "task")
	base.ToolManuals = []artifact.ID{testutil.ArtifactID(t, artifact.KindRecipe, "manual")}
	base.InvocationEffects = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "effect")}
	base.Obligations = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "obligation")}
	base.Resolutions = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "resolution")}
	base.WorkspaceClaims = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "claim")}
	base.Attempts = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "attempt")}
	base.BudgetCharges = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "charge")}
	base.Terminal = OutcomeSucceeded
	first, err := NewAgentTrajectory(base)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewAgentTrajectory(base)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || len(first.Lineage()) == 0 || first.Events[1].Branch != "candidate/a" {
		t.Fatalf("trajectory = %+v", first)
	}
}
