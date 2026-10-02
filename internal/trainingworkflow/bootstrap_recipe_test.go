package trainingworkflow

import (
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

// TestBootstrapRecordsModelLocation pins the discoverability contract:
// bootstrapping a recipe from a weights file records that file's location
// for the identified model, so every bootstrapped activation resolves to
// its bytes without a separate intake; the rerun keeps the recorded fact.
func TestBootstrapRecordsModelLocation(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.gguf")
	datasetPath := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(modelPath, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("training data"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := t.Context()
	if _, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath); err != nil {
		t.Fatal(err)
	}
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("weights"))
	if err != nil {
		t.Fatal(err)
	}
	recorded, err := artifact.AvailablePath(ctx, store, modelID, artifact.LocationFile)
	if err != nil {
		t.Fatalf("bootstrapped model has no recorded weights location: %v", err)
	}
	data, err := os.ReadFile(recorded)
	if err != nil || string(data) != "weights" {
		t.Fatalf("recorded location %s = (%q, %v), want the bootstrapped weights", recorded, data, err)
	}
	if _, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath); err != nil {
		t.Fatalf("repeat bootstrap with a recorded location: %v", err)
	}
}

// TestBootstrapVerifiesWithARealRun holds a training recipe's activation to
// a run that actually trained under it. Bootstrap publishes the recipe as a
// candidate and records no verification; a run whose code revision cannot
// be named trains but leaves it a candidate; a run at a named revision
// activates it on a gate bound to that revision and the environment the
// session observed, whose run depends on the session's observation and the
// checkpoint it published; the next run reverifies it on its own. No
// production file of the package imports the test fixture that once
// supplied a fabricated verification.
func TestBootstrapVerifiesWithARealRun(t *testing.T) {
	sources, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(sources.Imports, "overgo/internal/modelrecipetest") {
		t.Fatal("production training workflow imports the modelrecipetest fixture")
	}
	root := t.TempDir()
	modelPath := writeTrainingGGUF(t)
	datasetPath := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(datasetPath, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	recipeID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	modelID, err := identifyModel(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	status := func() recipe.Status {
		t.Helper()
		state, _, err := modelrecipe.Status(ctx, store, recipeID)
		if err != nil {
			t.Fatal(err)
		}
		return state
	}
	if state := status(); state != recipe.StatusCandidate {
		t.Fatalf("bootstrapped recipe is %q, want a candidate no run has verified", state)
	}
	train := func(output string) Result {
		t.Helper()
		result, err := Execute(ctx, Request{
			Repository: store, Recipe: recipeID, Observations: store,
			ModelDirectory: modelPath, DatasetPath: datasetPath, OutputDirectory: filepath.Join(root, output),
			Steps: 1, MaximumSequence: 16, Host: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	prior := executableCodeCommit
	executableCodeCommit = func(string) (string, error) { return "", errors.New("the executable names no code revision") }
	t.Cleanup(func() { executableCodeCommit = prior })
	train("unnamed")
	if state := status(); state != recipe.StatusCandidate {
		t.Fatalf("a run at no named revision left the recipe %q, want it a candidate", state)
	}
	for index, commit := range []string{strings.Repeat("c0", 20), strings.Repeat("d1", 20)} {
		stubExecutableCodeCommit(t, commit)
		result := train(fmt.Sprintf("run-%d", index))
		activation, found, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskTraining)
		if err != nil || !found || activation.Definition.ID != recipeID {
			t.Fatalf("run %d activation = (%s, found %t, %v), want %s active", index, activation.Definition.ID, found, err, recipeID)
		}
		verified, err := runrecord.VerifyEvidence(ctx, store, recipeID, activation.Event.Evidence)
		if err != nil {
			t.Fatal(err)
		}
		observation, err := runrecord.RequireServingObservation(ctx, store, result.Observation)
		if err != nil {
			t.Fatal(err)
		}
		if verified.Gate.CodeCommit != commit || verified.Run.CodeCommit != commit || verified.Gate.Environment != observation.Environment ||
			!slices.Contains(verified.Run.Inputs, result.Observation) || !slices.Contains(verified.Run.Inputs, result.Checkpoint.ID()) || verified.Run.MeasuredNS == 0 {
			t.Fatalf("run %d verification = gate %+v run %+v, want it bound to commit %s, environment %s, observation %s and checkpoint %s",
				index, verified.Gate, verified.Run, commit, observation.Environment, result.Observation, result.Checkpoint.ID())
		}
	}
}

// TestBootstrapGroundsAStoredObjectivesReferences holds a bootstrap to every
// identity the recipe's run plan resolves. A store rebuilt from one that
// never declared an objective's dataset and split holds the objective and its
// published recipe without them, as the live store did for the dit, mot and
// hybrid routes; the rerun of the bootstrap that opens a route's session
// declares them, so the run resolves its objective when its checkpoint
// publishes rather than failing there after it trained.
func TestBootstrapGroundsAStoredObjectivesReferences(t *testing.T) {
	t.Parallel()
	root, ctx := t.TempDir(), t.Context()
	modelPath, data := filepath.Join(root, "model.gguf"), []byte("training data")
	if err := os.WriteFile(modelPath, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	source, err := overgodb.Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	recipeID, err := BootstrapRecipe(ctx, source, modelPath, data, TokenContract)
	if err != nil {
		t.Fatal(err)
	}
	dataset, split, _, err := TextAuthority(data)
	if err != nil {
		t.Fatal(err)
	}
	rebuilt, err := overgodb.Open(filepath.Join(root, "rebuilt"))
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	_, head := source.Head()
	for sequence := uint64(1); sequence <= head; sequence++ {
		delta, _, err := source.CommitDeltaAt(ctx, sequence)
		if err != nil {
			t.Fatal(err)
		}
		batch := delta.Delta
		batch.Key = fmt.Sprintf("rebuild/%06d", sequence)
		batch.Artifacts = slices.DeleteFunc(batch.Artifacts, func(descriptor artifact.Descriptor) bool {
			return descriptor.ID == dataset || descriptor.ID == split
		})
		if _, err := rebuilt.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	definition, err := recipe.RequireDefinition(ctx, rebuilt, recipeID)
	if err != nil {
		t.Fatal(err)
	}
	objective, _ := definition.PrimaryDependency(recipe.DependencyObjective)
	if _, err := trainingprogram.CompileObjectiveMatrix(ctx, rebuilt, []artifact.ID{objective}); err == nil {
		t.Fatal("the rebuilt store resolves an objective whose dataset and split it never declared")
	}
	if again, err := BootstrapRecipe(ctx, rebuilt, modelPath, data, TokenContract); err != nil || again != recipeID {
		t.Fatalf("rerun bootstrap = (%s, %v), want the published recipe %s", again, err, recipeID)
	}
	if _, err := trainingprogram.CompileObjectiveMatrix(ctx, rebuilt, []artifact.ID{objective}); err != nil {
		t.Fatalf("after the rerun bootstrap its recipe's objective still does not resolve: %v", err)
	}
}

func TestBootstrapTokenRecipeBindsStoredOptimizerPolicy(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.gguf")
	datasetPath := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("training data"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := t.Context()
	definitionID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	content, ok, err := artifact.ReadContent(ctx, store, definitionID)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("recipe definition %s was not stored", definitionID)
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	optimizerID, ok := definition.PrimaryDependency(recipe.DependencyOptimizer)
	if !ok {
		t.Fatal("bootstrap recipe has no optimizer policy dependency")
	}
	if optimizerID != trainingprogram.BuiltinOptimizerPolicy().ID {
		t.Fatalf("optimizer policy = %s, want %s", optimizerID, trainingprogram.BuiltinOptimizerPolicy().ID)
	}
	if _, err := trainingprogram.RequireOptimizerPolicy(ctx, store, optimizerID); err != nil {
		t.Fatalf("load optimizer policy: %v", err)
	}

	repeatedID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatalf("repeat bootstrap: %v", err)
	}
	if repeatedID != definitionID {
		t.Fatalf("repeat recipe = %s, want %s", repeatedID, definitionID)
	}
}
