package evaluation

import (
	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"testing"
)

func TestAgentTrajectoryEvaluationPlan(t *testing.T) {
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	base, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "agent-model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "agent-recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "agent-environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	trajectory := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")
	judge := testutil.ArtifactID(t, artifact.KindProfile, "judge-v1")
	plan, err := BindAgentTrajectoryPlan(base, []artifact.ID{trajectory}, []artifact.ID{judge})
	if err != nil {
		t.Fatal(err)
	}
	authorities, found := plan.AgentTrajectoryAuthorities()
	if !found || len(authorities.Deterministic) != len(requiredAgentTrajectoryScores) || authorities.AdvisoryJudges[0] != judge {
		t.Fatalf("agent plan = %+v", authorities)
	}
	content, err := plan.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParsePlan(content.Data)
	if err != nil || parsed.Identity() != plan.Identity() {
		t.Fatalf("round trip = %s %v", parsed.Identity(), err)
	}
	if _, err := BindAgentTrajectoryPlan(base, nil, nil); err == nil {
		t.Fatal("trajectory-free agent plan admitted")
	}
}
