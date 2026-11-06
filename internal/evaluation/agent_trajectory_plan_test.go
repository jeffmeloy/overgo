package evaluation

import (
	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"testing"
)

func TestAgentTrajectoryEvaluationPlan(t *testing.T) {
	t.Parallel()
	exact := testcheck.Must(CompileExact(exactFixture()))(t)
	base := testcheck.Must(BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "agent-model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "agent-recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "agent-environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}))(t)
	trajectory := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")
	judge := testutil.ArtifactID(t, artifact.KindProfile, "judge-v1")
	plan := testcheck.Must(BindAgentTrajectoryPlan(base, []artifact.ID{trajectory}, []artifact.ID{judge}))(t)
	authorities := plan.body.Agent
	if authorities == nil || len(authorities.Deterministic) != len(requiredAgentTrajectoryScores) || authorities.AdvisoryJudges[0] != judge {
		t.Fatalf("agent plan = %+v", authorities)
	}
	content := testcheck.Must(plan.Content())(t)
	parsed, err := ParsePlan(content.Data)
	if err != nil || parsed.Identity() != plan.Identity() {
		t.Fatalf("round trip = %s %v", parsed.Identity(), err)
	}
	if _, err := BindAgentTrajectoryPlan(base, nil, nil); err == nil {
		t.Fatal("trajectory-free agent plan admitted")
	}
}
