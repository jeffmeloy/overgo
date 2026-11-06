package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAReadingFromAnUnverifiedModelSaysSo: a recipe's row reads what was attempted while something stood behind its
// model, and the verification that activated it, whose attempts ran before it could; an attempt made before, a record
// that says nothing, and the two states that stand each keep their place.
func TestAReadingFromAnUnverifiedModelSaysSo(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	model := testutil.ArtifactID(t, artifact.KindModel, "attempted-model")
	profile := testutil.ArtifactID(t, artifact.KindProfile, "attempted-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "attempted-definition")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "attempted-environment")
	stray := testutil.ArtifactID(t, artifact.KindRun, "a run that proved nothing")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/attempted", Artifacts: []artifact.Descriptor{
		{ID: model}, {ID: profile}, {ID: definitionID}, {ID: environment}, {ID: stray},
	}}); err != nil {
		t.Fatal(err)
	}
	definition := testcheck.Must(modelrecipe.InferenceWithModelDefinition(
		model, profile, definitionID, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/attempted/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, "fixture/attempted/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "fixture/attempted/evidence", definition.ID))(t)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/attempted/active", definition, verification, recipe.EvidenceVerified, "attempted fixture", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	publish := func(name, state string, run artifact.ID) {
		t.Helper()
		observation := testcheck.Must(runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: definition.ID, Environment: environment, Task: recipe.TaskInference, Run: run,
			Verification: state, Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 1, MeasuredNS: 1,
		}))(t)
		batch := testcheck.Must(observation.Batch("fixture/attempted/" + name))(t)
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	failures := func() int {
		t.Helper()
		summaries := testcheck.Must(servingByRecipe(ctx, store))(t)
		summary := summaries[definition.ID]
		if summary == nil {
			return 0
		}
		return summary.failures
	}
	notVerified, broken := string(modelrecipe.StateNotVerified), string(modelrecipe.StateBroken)
	for _, test := range []struct {
		name, state string
		run         artifact.ID
		counts      bool
	}{
		{"before it was verified", notVerified, artifact.ID{}, false},
		{"while its verification was broken", broken, stray, false},
		{"as the run that verified it", notVerified, verification.Run, true},
		{"after it was verified", string(modelrecipe.StateVerified), artifact.ID{}, true},
		{"with its evidence out of date", string(modelrecipe.StateOutOfDate), artifact.ID{}, true},
		{"before records held a state", "", artifact.ID{}, true},
	} {
		before := failures()
		publish(test.name, test.state, test.run)
		if counted := failures() > before; counted != test.counts {
			t.Fatalf("an attempt %s: counted %t, want %t", test.name, counted, test.counts)
		}
	}
}
