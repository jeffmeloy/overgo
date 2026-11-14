package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnUnloadableRecipeIsNotRunnable: a candidate's closure answers for what would run. An inference recipe with no forward
// stage passes the module graph and cannot load, so it is not runnable, and a representation bridge, whose stages only this
// package's catalog holds, is.
func TestAnUnloadableRecipeIsNotRunnable(t *testing.T) {
	t.Parallel()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "unloadable")
	whole := testcheck.Must(inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest))(t)
	if closure := ValidateCandidateRecipe(whole); !closure.Runnable {
		t.Fatalf("a whole recipe: %+v", closure)
	}
	compile, decode := whole.Nodes[0], whole.Nodes[1]
	partial := testcheck.Must(recipe.NewDefinitionWithDependencies(
		recipe.TaskInference, whole.Dependencies, []recipe.Node{compile, decode},
		[]recipe.Edge{{From: recipe.Endpoint{Node: compile.ID, Port: "plan"}, To: recipe.Endpoint{Node: decode.ID, Port: "plan"}}},
		nil, []recipe.Output{{Name: "session", Data: recipe.DataSessionPlan, Source: recipe.Endpoint{Node: decode.ID, Port: "session"}}},
	))(t)
	if closure := ValidateCandidateRecipe(partial); closure.Runnable {
		t.Fatalf("a recipe with no forward stage reads as runnable: %+v", closure)
	}
	bridge := testcheck.Must((RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementDevice, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyDeviceF32, TargetResidency: recipe.ResidencyDeviceF32,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(modelID, testutil.ArtifactID(t, artifact.KindModel, "target"), testutil.ArtifactID(t, artifact.KindProfile, "source contract"),
		testutil.ArtifactID(t, artifact.KindProfile, "target contract"), testutil.ArtifactID(t, artifact.KindAdapter, "bridge")))(t)
	if closure := ValidateCandidateRecipe(bridge); !closure.Runnable {
		t.Fatalf("a representation bridge: %+v", closure)
	}
}
