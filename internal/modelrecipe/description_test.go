package modelrecipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modeltest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestRuntimeDescriptionPreservesCompiledOrderAndEvidence(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "description-model")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "description-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "description-definition")
	definition, err := InferenceWithModelDefinition(
		modelID, profileID, definitionID, recipe.PlacementHost,
		DecodeSessionRequest, recipe.ResidencyHostCache,
	)
	testcheck.NoError(t, err)
	fixture := modeltest.Qwen35DenseRecurrentPair()
	plan, err := compileInferenceFixture(definition, fixture.Spec, fixture.ServingWeights())
	testcheck.NoError(t, err)
	plan.Evidence = []artifact.ID{
		testutil.ArtifactID(t, artifact.KindEvidence, "description-gate"),
		testutil.ArtifactID(t, artifact.KindRun, "description-run"),
	}
	description, err := Describe(plan)
	testcheck.NoError(t, err)
	compiled, err := recipe.CompileProgram(definition, catalog)
	testcheck.NoError(t, err)
	want := compiled.Stages()
	if len(description.Stages) != len(want) {
		t.Fatalf("stage count = %d, want %d", len(description.Stages), len(want))
	}
	for index := range want {
		if description.Stages[index].Node.ID != want[index].Node.ID {
			t.Fatalf("stage %d = %s, want %s", index, description.Stages[index].Node.ID, want[index].Node.ID)
		}
	}
	if description.CacheIdentity != definition.ID || !slices.Equal(description.Evidence, plan.Evidence) {
		t.Fatalf("description identity/evidence = %+v", description)
	}
}
