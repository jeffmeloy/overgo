package evaluation

import (
	"testing"

	"overgo/internal/artifact"
)

const planTestCommit = "0123456789abcdef0123456789abcdef01234567"

func TestPlanBindsEveryEvaluationAuthority(t *testing.T) {
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	base := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan, err := BindExact(exact, base)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Identity().Kind() != artifact.KindProfile ||
		plan.body.Dataset != exact.dataset || plan.body.Split != exact.split ||
		plan.body.CaseProfile != exact.Identity() || plan.body.Scorer.Kind() != artifact.KindProfile ||
		plan.body.Execution.Kind() != artifact.KindProfile {
		t.Fatalf("plan authorities = %+v", plan.body)
	}

	variants := []ExactAuthorities{base, base, base, base, base}
	variants[0].ModelDefinition = planID(t, artifact.KindModelDefinition, "other-model")
	variants[1].RuntimeRecipe = planID(t, artifact.KindRecipe, "other-recipe")
	variants[2].CodeCommit = "abcdef0123456789abcdef0123456789abcdef01"
	variants[3].Environment = planID(t, artifact.KindEvidence, "other-environment")
	variants[4].Execution.Lifecycle = LifecycleIsolated
	for index, variant := range variants {
		other, err := BindExact(exact, variant)
		if err != nil {
			t.Fatal(err)
		}
		if other.Identity() == plan.Identity() {
			t.Fatalf("authority %d did not change identity", index)
		}
	}

	changed := exactFixture()
	changed.Cases[0].Prompt = "other prompt"
	otherExact, err := CompileExact(changed)
	if err != nil {
		t.Fatal(err)
	}
	other, err := BindExact(otherExact, base)
	if err != nil {
		t.Fatal(err)
	}
	if other.Identity() == plan.Identity() || other.body.Dataset == plan.body.Dataset ||
		other.body.Split == plan.body.Split || other.body.CaseProfile == plan.body.CaseProfile {
		t.Fatal("case, dataset, or split authority did not change identity")
	}
}

func TestPlanRejectsInvalidEvaluationAuthority(t *testing.T) {
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	valid := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	invalid := []ExactAuthorities{valid, valid, valid, valid, valid}
	invalid[0].ModelDefinition = artifact.ID{}
	invalid[1].RuntimeRecipe = planID(t, artifact.KindProfile, "recipe")
	invalid[2].CodeCommit = "revision"
	invalid[3].Environment = planID(t, artifact.KindProfile, "environment")
	invalid[4].Execution.Lifecycle = "automatic"
	for index, authorities := range invalid {
		if _, err := BindExact(exact, authorities); err == nil {
			t.Fatalf("invalid authority %d accepted", index)
		}
	}
}

func planID(t testing.TB, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
