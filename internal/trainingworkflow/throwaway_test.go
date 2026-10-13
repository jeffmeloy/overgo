package trainingworkflow

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// requireThrowaway holds the checkpoint in directory to what the store keeps
// of a throw-away one: its document, which a session's record and a verdict
// name, with what it descends from, and nothing that says where its files
// are, so no catalog lists files its run will remove. A checkpoint that holds a trained GGUF model has not
// registered that model either. It returns the checkpoint.
func requireThrowaway(t *testing.T, store *overgodb.Store, directory string) trainingprogram.Checkpoint {
	t.Helper()
	ctx := t.Context()
	checkpoint, err := trainingprogram.LoadCheckpoint(directory)
	if err != nil {
		t.Fatalf("%s: %v", directory, err)
	}
	if locations, err := store.Locations(ctx, checkpoint.ID()); err != nil || len(locations) != 0 {
		t.Fatalf("the store records the throw-away checkpoint of %s at %v (%v), want no location", directory, locations, err)
	}
	if _, found, err := artifact.ReadContent(ctx, store, checkpoint.ID()); err != nil || !found {
		t.Fatalf("the store holds the document of the throw-away checkpoint of %s: %t (%v), want it held", directory, found, err)
	}
	parents, err := store.Parents(ctx, checkpoint.ID())
	for _, edge := range checkpoint.ArtifactLineage() {
		if err != nil || !slices.Contains(parents, edge) {
			t.Fatalf("the store holds the parents %v of the throw-away checkpoint of %s (%v), want what it descends from: %v", parents, directory, err, checkpoint.ArtifactLineage())
		}
	}
	model := filepath.Join(directory, trainingprogram.CheckpointModel)
	if _, err := os.Stat(model); err != nil {
		return checkpoint
	}
	inventory, err := modelartifact.FromGGUFPath(model)
	if err != nil {
		t.Fatal(err)
	}
	if _, held, err := store.Artifact(ctx, inventory.Manifest.ID); err != nil || held {
		t.Fatalf("the store registers the trained model of the throw-away checkpoint of %s: %t (%v), want it not registered", directory, held, err)
	}
	return checkpoint
}

// receiptLines reads a receipt: one line a directory, its verb, bytes, place,
// path and rule between tabs. It returns the lines by path.
func receiptLines(t *testing.T, receipt string) map[string][]string {
	t.Helper()
	lines := map[string][]string{}
	for line := range strings.Lines(receipt) {
		fields := strings.Split(strings.TrimSuffix(line, "\n"), "\t")
		if len(fields) != 5 || lines[fields[3]] != nil {
			t.Fatalf("receipt line %q has %d fields, or is its path's second line: want verb, bytes, place, path and rule, once a path", line, len(fields))
		}
		if _, err := strconv.ParseInt(fields[1], 10, 64); err != nil {
			t.Fatalf("receipt line %q gives its bytes as %q", line, fields[1])
		}
		lines[fields[3]] = fields
	}
	return lines
}

// receiptVerb is what a receipt says was done with path, and nothing when it
// has no line for it.
func receiptVerb(lines map[string][]string, path string) string {
	if line := lines[path]; line != nil {
		return line[0]
	}
	return ""
}

// unlisted is a store whose listing does not come whole: a query of it ends
// with err, or returns cut and no more. Every other read is the store's own.
type unlisted struct {
	*overgodb.Store
	cut overgodb.QueryResult
	err error
}

// Query returns what the listing is set to end with, and reads nothing.
func (store unlisted) Query(context.Context, overgodb.Query) (overgodb.QueryResult, error) {
	return store.cut, store.err
}

// TestExperimentRemovesItsCheckpointsOnceItsRecordIsKept holds what an
// experiment's intermediate checkpoints are to the store, and how they are
// removed.
//
// A session asked to treat its checkpoint as throw-away records the
// checkpoint's document and nothing else: no location, and for a GGUF base
// no trained model and no candidate. That is so for token sessions over a
// GGUF, the layout the pilot's checkpoints have, and for a reward session, the
// pilot's kind. A session still resumes from such a checkpoint, and a reward
// checkpoint is still held to be its session's, because each of those reads
// the directory or the document and not a location. A reward session trained
// again in another directory publishes the checkpoint it published before,
// which the store records once. A session not asked records as before.
//
// RemoveCheckpoints removes the directories it is given that hold a
// checkpoint, and says so a line each. One the store records at that very
// path is kept, a copy of it elsewhere is not, a directory holding no
// checkpoint is passed over, and a folder left empty goes too. Given a
// reason it removes nothing and writes the reason against each directory.
//
// A directory in which the store records a registered model is kept too,
// whatever its checkpoint is to the store, with a line that names the model,
// while a directory beside it with the same files goes. And when the store's
// models cannot be listed, or the listing is cut short, nothing is removed:
// a listing that did not come whole is not a store that registers no model.
func TestExperimentRemovesItsCheckpointsOnceItsRecordIsKept(t *testing.T) {
	t.Parallel()
	t.Run("token sessions over a GGUF, and their removal", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		root := t.TempDir()
		modelPath, datasetPath := writeTrainingGGUF(t), filepath.Join(root, "dataset.txt")
		if err := os.WriteFile(datasetPath, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(filepath.Join(root, "store"))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
			t.Fatal(err)
		}
		recipeID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		request := Request{
			Repository: store, Recipe: recipeID, Observations: store,
			ModelDirectory: modelPath, DatasetPath: datasetPath, MaximumSequence: 16, Host: true, throwaway: true,
		}
		models := func() int {
			registered, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindModel, MaxResults: 100, Projection: overgodb.ProjectArtifacts})
			if err != nil || registered.Truncated {
				t.Fatal(err, registered.Truncated)
			}
			return len(registered.Artifacts)
		}
		// One session is recorded as before, of more steps than any below, so its checkpoint is none of theirs.
		request.throwaway, request.Steps, request.OutputDirectory = false, 3, filepath.Join(root, "holds", "recorded")
		recorded, err := Execute(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		inventory, err := modelartifact.FromGGUFPath(filepath.Join(request.OutputDirectory, trainingprogram.CheckpointModel))
		if err != nil {
			t.Fatal(err)
		}
		locations, err := store.Locations(ctx, recorded.Checkpoint.ID())
		if _, held, heldErr := store.Artifact(ctx, inventory.Manifest.ID); err != nil || heldErr != nil || !held || len(locations) == 0 || !recorded.Candidate.Valid() {
			t.Fatalf("a session not asked to throw its checkpoint away recorded candidate %s, locations %v and its trained model %t (%v, %v), want all three", recorded.Candidate, locations, held, err, heldErr)
		}
		before := models()
		thrown := filepath.Join(root, "thrown")
		request.throwaway = true
		uninterrupted := verifyWorkflowResume(t, thrown, request)
		if uninterrupted.Candidate.Valid() {
			t.Fatalf("a throw-away session returned candidate %s, want none", uninterrupted.Candidate)
		}
		var directories []string
		for _, name := range []string{"uninterrupted", "first", "resumed"} {
			directories = append(directories, filepath.Join(thrown, name))
			requireThrowaway(t, store, directories[len(directories)-1])
		}
		if after := models(); after != before {
			t.Fatalf("three throw-away sessions over a GGUF left the store with %d models, and it held %d before them", after, before)
		}

		// The removal: the three, the recorded one, a copy of the recorded one, and a folder that holds no checkpoint.
		held, copied, empty := filepath.Join(root, "holds", "recorded"), filepath.Join(root, "holds", "copied"), filepath.Join(root, "empty")
		if err := errors.Join(os.CopyFS(copied, os.DirFS(held)), os.Mkdir(empty, 0o700)); err != nil {
			t.Fatal(err)
		}
		// And two more throw-away directories, each a copy of the first: a model is registered by hand from a file of one.
		prototype, sibling := filepath.Join(root, "models", "prototype"), filepath.Join(root, "models", "sibling")
		if err := errors.Join(os.CopyFS(prototype, os.DirFS(directories[0])), os.CopyFS(sibling, os.DirFS(directories[0]))); err != nil {
			t.Fatal(err)
		}
		byHand, err := modelartifact.FromGGUFPath(filepath.Join(prototype, trainingprogram.CheckpointModel))
		if err != nil {
			t.Fatal(err)
		}
		registration, err := byHand.Batch("fixture/a model registered by hand")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, registration); err != nil {
			t.Fatal(err)
		}
		// A model whose bytes are with a provider is recorded at an address no directory holds, which is no path to look under.
		served := testutil.ArtifactID(t, artifact.KindModel, "a model a provider serves")
		if _, err := artifact.Publish(ctx, store, artifact.Batch{
			Key: "fixture/a model a provider serves", Artifacts: []artifact.Descriptor{{ID: served}},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: served, Kind: artifact.LocationRemote, Value: "https://provider.invalid/v1/models/served"}, Action: artifact.LocationAdd}},
		}); err != nil {
			t.Fatal(err)
		}
		directories = append(directories, held, copied, empty, prototype, sibling)
		var receipt bytes.Buffer
		const reason = "kept to be read by hand"
		if err := RemoveCheckpoints(ctx, store, &receipt, reason, directories...); err != nil {
			t.Fatal(err)
		}
		lines := receiptLines(t, receipt.String())
		for _, directory := range directories {
			if _, err := os.Stat(directory); err != nil || receiptVerb(lines, directory) != "kept" || lines[directory][4] != reason {
				t.Fatalf("given a reason to keep, %s is %v with the line %q, want it standing and kept for %q", directory, err, lines[directory], reason)
			}
		}
		// A store whose models cannot be listed, or are listed cut short, is not a store that registers none: nothing goes.
		errListing := errors.New("the models cannot be listed")
		for name, listing := range map[string]unlisted{
			"cannot be listed":     {Store: store, err: errListing},
			"are listed cut short": {Store: store, cut: overgodb.QueryResult{Truncated: true}},
		} {
			receipt.Reset()
			err := RemoveCheckpoints(ctx, listing, &receipt, "", directories...)
			if err == nil || listing.err != nil && !errors.Is(err, errListing) || strings.Contains(receipt.String(), "removed\t") {
				t.Fatalf("with a store whose models %s the removal ended with %v and the lines %q, want it failed for that with nothing removed", name, err, receipt.String())
			}
			for _, directory := range directories {
				if _, err := os.Stat(directory); err != nil {
					t.Fatalf("with a store whose models %s, %s is %v, want it standing", name, directory, err)
				}
			}
		}
		receipt.Reset()
		inUse := directories[1]
		if runtime.GOOS == "windows" {
			// A checkpoint with a file held open is in use: it is left whole, the others still go, and the call says what failed.
			open, err := os.Open(filepath.Join(inUse, trainingprogram.CheckpointFilename))
			if err != nil {
				t.Fatal(err)
			}
			err = RemoveCheckpoints(ctx, store, &receipt, "", inUse, directories[0])
			if closeErr := open.Close(); err == nil || closeErr != nil {
				t.Fatalf("removing a checkpoint with a file held open = %v (close: %v), want an error", err, closeErr)
			}
			lines = receiptLines(t, receipt.String())
			if _, err := trainingprogram.LoadCheckpoint(inUse); err != nil || receiptVerb(lines, inUse) != "left" || receiptVerb(lines, directories[0]) != "removed" {
				t.Fatalf("with a file of %s held open it loads as %v and the lines are %q and %q, want it left whole and the other removed", inUse, err, lines[inUse], lines[directories[0]])
			}
			if receiptVerb(lines, thrown) != "kept" {
				t.Fatalf("the folder %s still holds checkpoints and its line is %q, want it kept", thrown, lines[thrown])
			}
			receipt.Reset()
			directories = directories[1:]
		}
		if err := RemoveCheckpoints(ctx, store, &receipt, "", directories...); err != nil {
			t.Fatal(err)
		}
		lines = receiptLines(t, receipt.String())
		for _, directory := range directories {
			line := lines[directory]
			_, err := os.Stat(directory)
			switch directory {
			case held:
				if _, err := trainingprogram.LoadCheckpoint(held); err != nil || line == nil || line[0] != "kept" || !strings.Contains(line[4], "the store records this checkpoint") {
					t.Fatalf("the checkpoint the store records at %s loads as %v with the line %q, want it whole and kept for the store's record of it", held, err, line)
				}
			case prototype:
				if _, err := trainingprogram.LoadCheckpoint(prototype); err != nil || line == nil || line[0] != "kept" || !strings.Contains(line[4], byHand.Manifest.ID.String()) || !strings.Contains(line[4], "never") {
					t.Fatalf("the directory %s, from a file of which model %s is registered, loads as %v with the line %q, want it whole and kept for that model, which is never a run's to throw away", prototype, byHand.Manifest.ID, err, line)
				}
			case empty:
				if err != nil || line != nil {
					t.Fatalf("the folder %s, which holds no checkpoint, is %v with the line %q, want it passed over", empty, err, line)
				}
			default:
				if !errors.Is(err, os.ErrNotExist) || line == nil || line[0] != "removed" || line[1] == "0" {
					t.Fatalf("the throw-away checkpoint at %s is %v with the line %q, want it removed with its bytes", directory, err, line)
				}
			}
		}
		if _, err := os.Stat(thrown); !errors.Is(err, os.ErrNotExist) || lines[thrown] != nil {
			t.Fatalf("the folder %s, left empty, is %v with the line %q, want it removed with no line", thrown, err, lines[thrown])
		}
		for _, parent := range []string{filepath.Dir(held), filepath.Dir(prototype)} {
			if receiptVerb(lines, parent) != "kept" {
				t.Fatalf("the folder %s still holds a checkpoint that was kept and its line is %q, want it kept", parent, lines[parent])
			}
		}
	})

	t.Run("a reward session", func(t *testing.T) {
		t.Parallel()
		root, request := rewardFixture(t)
		request.throwaway = true
		uninterrupted := verifyWorkflowResume(t, root, request)
		store := request.Observations.(*overgodb.Store)
		base, err := identifyModel(request.ModelDirectory)
		if err != nil {
			t.Fatal(err)
		}
		requireThrowaway(t, store, filepath.Join(root, "uninterrupted"))
		first := requireThrowaway(t, store, filepath.Join(root, "first"))
		resumed := requireThrowaway(t, store, filepath.Join(root, "resumed"))
		if err := errors.Join(request.Rewards.published(first, base, trainingprogram.Checkpoint{}, 1), request.Rewards.published(resumed, base, first, 2)); err != nil {
			t.Fatalf("a throw-away reward checkpoint, loaded again, is not held to be its session's: %v", err)
		}
		// The same session trained again in another directory publishes the
		// same checkpoint, whose document the store holds by then, as a run
		// begun again after its checkpoints were removed trains it.
		again := request
		again.Steps, again.OutputDirectory = uninterrupted.Plan.Updates(), filepath.Join(root, "again")
		if repeated, err := Execute(t.Context(), again); err != nil || repeated.Checkpoint.ID() != uninterrupted.Checkpoint.ID() {
			t.Fatalf("the uninterrupted session trained again in another directory published %s (%v), want the checkpoint %s it published before", repeated.Checkpoint.ID(), err, uninterrupted.Checkpoint.ID())
		}
		requireThrowaway(t, store, again.OutputDirectory)
	})
}
