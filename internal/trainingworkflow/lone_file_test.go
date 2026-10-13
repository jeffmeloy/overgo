package trainingworkflow

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
)

// TestARunOverAFileOfALargerModelRetiresTheFilesOwnActivation holds what a run does with a
// training activation that one file of its weights holds as a model of its own.
//
// A weights file alone in its directory is a model. A recipe is bootstrapped for it, and a
// session under the recipe verifies and activates it. The model then gains a second file and an
// index, so a run given the first file binds the model of both files. The first file's own
// activation names a model no run binds any more, and its proof is refused, with both models
// named in the refusal.
//
// The next run given that file retires the activation as orphaned. It does not while the tree's
// code revision cannot be named, and it does not while another live file of that model is still
// bound as that model. Once retired, the recipe is superseded and refused, the refusal names
// what a run binds from the file, the model has no active training recipe, and a run after it
// finds nothing to retire.
func TestARunOverAFileOfALargerModelRetiresTheFilesOwnActivation(t *testing.T) {
	// Serial: it replaces how a run names its code revision.
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	shard := func(directory, file, tensor string, value float32) string {
		t.Helper()
		if err := os.MkdirAll(filepath.Join(root, directory), 0o700); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(root, directory, file)
		if err := safetensors.Save(path, map[string][]float32{tensor: {value}}, map[string][]int{tensor: {1}}, nil); err != nil {
			t.Fatal(err)
		}
		return path
	}
	first := shard("model", "model-00001-of-00002.safetensors", "first", 1)

	stubExecutableCodeCommit(t, strings.Repeat("c0", 20))
	lone, err := identifyAndRecordWeights(ctx, store, first)
	if err != nil {
		t.Fatal(err)
	}
	recipeID, err := BootstrapRecipe(ctx, store, first, []byte("the text a run reads"), TokenContract)
	if err != nil {
		t.Fatal(err)
	}
	observer, err := NewObserver(store, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := observer.Admit(ctx, lone, recipeID); err != nil {
		t.Fatal(err)
	}
	observation, err := observer.Finish(ctx, lone, recipeID, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := observer.Verify(ctx, recipeID, observation); err != nil {
		t.Fatal(err)
	}
	// active says whether the lone file's model has an active training recipe, and which.
	active := func() (artifact.ID, bool) {
		t.Helper()
		id, bound, err := artifact.ResolveAlias(ctx, store, modelrecipe.ActiveAliasPrefix(recipe.TaskTraining)+lone.String())
		if err != nil {
			t.Fatal(err)
		}
		return id, bound
	}
	if id, bound := active(); !bound || id != recipeID {
		t.Fatalf("the lone file's model has the active training recipe %s (%t), want the bootstrapped %s", id, bound, recipeID)
	}

	// The model gains a second file and an index: the first file is now a part of it.
	shard("model", "model-00002-of-00002.safetensors", "second", 2)
	if err := os.WriteFile(filepath.Join(root, "model", "model.safetensors.index.json"),
		[]byte(`{"weight_map":{"first":"model-00001-of-00002.safetensors","second":"model-00002-of-00002.safetensors"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	whole, err := WeightsIdentity(first)
	if err != nil || whole == lone {
		t.Fatalf("a run over the first of two files binds %s (%v), want a model other than the file's own %s", whole, err, lone)
	}
	if _, err := ResolveSessionAuthority(ctx, store, whole, recipeID); err == nil || !strings.Contains(err.Error(), lone.String()) || !strings.Contains(err.Error(), whole.String()) {
		t.Fatalf("a run of the whole model under the lone file's recipe answered %v, want it refused, naming the recipe's model %s and the run's %s", err, lone, whole)
	}

	// run is a run over the first file: it binds the whole model, whatever it retires.
	run := func() {
		t.Helper()
		if bound, err := identifyAndRecordWeights(ctx, store, first); err != nil || bound != whole {
			t.Fatalf("a run over the first file bound %s (%v), want %s", bound, err, whole)
		}
	}
	prior := executableCodeCommit
	executableCodeCommit = func(string) (string, error) { return "", errors.New("the tree has uncommitted changes") }
	run()
	executableCodeCommit = prior
	if _, bound := active(); !bound {
		t.Fatal("a run from a tree whose revision cannot be named retired the activation, and no claim is recorded from such a tree")
	}

	// A copy of the file alone in another directory is still that model to a run, so the
	// activation is not orphaned.
	copied := shard("copy", "model.safetensors", "first", 1)
	if again, err := identifyAndRecordWeights(ctx, store, copied); err != nil || again != lone {
		t.Fatalf("the file's copy, alone in its directory, is bound as %s (%v), want the file's own model %s", again, err, lone)
	}
	run()
	if _, bound := active(); !bound {
		t.Fatal("the activation was retired while a live file of its model is still bound as that model")
	}
	if err := os.Remove(copied); err != nil {
		t.Fatal(err)
	}

	run()
	if id, bound := active(); bound {
		t.Fatalf("after a run over the file the lone file's model still has the active training recipe %s", id)
	}
	eligibility, err := modelrecipe.RecipeEligibility(ctx, store, recipeID)
	if err != nil || eligibility.Status != recipe.StatusSuperseded || !strings.Contains(eligibility.Refusal, whole.String()) || !strings.Contains(eligibility.Refusal, loneFileReason) {
		t.Fatalf("the lone file's recipe is %q, refused as %q (%v), want it superseded, with the reason and the model a run binds, %s", eligibility.Status, eligibility.Refusal, err, whole)
	}
	if _, err := ResolveSessionAuthority(ctx, store, lone, recipeID); err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("a run under the retired recipe answered %v, want it refused", err)
	}
	// A run after it finds nothing to retire.
	run()
}
