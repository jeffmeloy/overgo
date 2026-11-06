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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestBootstrapRecordsModelLocation pins the discoverability contract:
// bootstrapping a recipe from a weights file records that file's location
// for the identified model, so every bootstrapped activation resolves to
// its bytes without a separate intake; the rerun keeps the recorded fact.
// TestBootstrapKeepsRecordedFileFacts records a pipeline in a store that already
// holds its source file as plain text. A file's identity is its bytes and its
// media type a fact the store keeps, so the run goes on and the fact stands.
func TestBootstrapKeepsRecordedFileFacts(t *testing.T) {
	t.Parallel()
	root := testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	testcheck.NoError(t, os.Remove(filepath.Join(root, "reference.safetensors")))
	source := []byte("class MOSSImagePipeline:\n    pass\n")
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "pipeline_moss_image.py"), source, 0o600))
	store := overgodbtest.Open(t, t.TempDir())
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, source))(t)
	const recorded = "text/plain"
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "registered", Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(source)), MediaType: recorded}}}); err != nil {
		t.Fatal(err)
	}
	model := testcheck.Must(identifyAndRecordWeights(t.Context(), store, root))(t)
	if _, found, err := store.Manifest(t.Context(), model); err != nil || !found {
		t.Fatalf("the pipeline was not recorded: %v", err)
	}
	if descriptor, _, err := store.Artifact(t.Context(), id); err != nil || descriptor.MediaType != recorded {
		t.Fatalf("source file fact %+v, %v", descriptor, err)
	}
}

func TestBootstrapRecordsModelLocation(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.safetensors")
	datasetPath := filepath.Join(root, "dataset.txt")
	testcheck.NoError(t, os.WriteFile(modelPath, []byte("weights"), 0o600))
	testcheck.NoError(t, os.WriteFile(datasetPath, []byte("training data"), 0o600))
	store := overgodbtest.Create(t, filepath.Join(root, "repodb"))

	ctx := t.Context()
	if _, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath); err != nil {
		t.Fatal(err)
	}
	modelID := testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("weights")))(t)
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
	sources := testcheck.Must(build.ImportDir(".", 0))(t)
	if slices.Contains(sources.Imports, "overgo/internal/modelrecipetest") {
		t.Fatal("production training workflow imports the modelrecipetest fixture")
	}
	root := t.TempDir()
	modelPath := writeTrainingGGUF(t)
	datasetPath := filepath.Join(root, "dataset.txt")
	testcheck.NoError(t, os.WriteFile(datasetPath, bytes.Repeat([]byte("a b c d "), 64), 0o600))
	store := overgodbtest.Create(t, filepath.Join(root, "store"))
	ctx := t.Context()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	recipeID := testcheck.Must(BootstrapTokenRecipe(ctx, store, modelPath, datasetPath))(t)
	modelID := testcheck.Must(identifyModel(modelPath))(t)
	status := func() recipe.Status {
		t.Helper()
		state, _, err := modelrecipe.Status(ctx, store, recipeID)
		testcheck.NoError(t, err)
		return state
	}
	if state := status(); state != recipe.StatusCandidate {
		t.Fatalf("bootstrapped recipe is %q, want a candidate no run has verified", state)
	}
	train := func(output string) Result {
		t.Helper()
		result := testcheck.Must(Execute(ctx, Request{
			Repository: store, Recipe: recipeID, Observations: store,
			ModelDirectory: modelPath, DatasetPath: datasetPath, OutputDirectory: filepath.Join(root, output),
			Steps: 1, MaximumSequence: 16, Host: true,
		}))(t)
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
		verified := testcheck.Must(runrecord.VerifyEvidence(ctx, store, recipeID, activation.Event.Evidence))(t)
		observation := testcheck.Must(runrecord.RequireServingObservation(ctx, store, result.Observation))(t)
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
	modelPath, data := filepath.Join(root, "model.safetensors"), []byte("training data")
	testcheck.NoError(t, os.WriteFile(modelPath, []byte("weights"), 0o600))
	source := overgodbtest.Create(t, filepath.Join(root, "source"))
	recipeID := testcheck.Must(BootstrapRecipe(ctx, source, modelPath, data, TokenContract))(t)
	dataset, split, _, err := TextAuthority(data)
	testcheck.NoError(t, err)
	rebuilt := overgodbtest.Create(t, filepath.Join(root, "rebuilt"))
	_, head := source.Head()
	for sequence := uint64(1); sequence <= head; sequence++ {
		delta, _, err := source.CommitDeltaAt(ctx, sequence)
		testcheck.NoError(t, err)
		batch := delta.Delta
		batch.Key = fmt.Sprintf("rebuild/%06d", sequence)
		batch.Artifacts = slices.DeleteFunc(batch.Artifacts, func(descriptor artifact.Descriptor) bool {
			return descriptor.ID == dataset || descriptor.ID == split
		})
		if _, err := rebuilt.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	definition := testcheck.Must(recipe.RequireDefinition(ctx, rebuilt, recipeID))(t)
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
	modelPath := filepath.Join(root, "model.safetensors")
	datasetPath := filepath.Join(root, "dataset.txt")
	testcheck.NoError(t, os.WriteFile(modelPath, []byte("model"), 0o600))
	testcheck.NoError(t, os.WriteFile(datasetPath, []byte("training data"), 0o600))
	store := overgodbtest.Create(t, filepath.Join(root, "repodb"))

	ctx := t.Context()
	definitionID := testcheck.Must(BootstrapTokenRecipe(ctx, store, modelPath, datasetPath))(t)
	content, ok, err := artifact.ReadContent(ctx, store, definitionID)
	testcheck.NoError(t, err)
	if !ok {
		t.Fatalf("recipe definition %s was not stored", definitionID)
	}
	definition := testcheck.Must(recipe.ParseDefinition(content.Data))(t)
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
