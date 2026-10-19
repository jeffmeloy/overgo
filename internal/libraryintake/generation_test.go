package libraryintake

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestMOSSGenerationRegistration(t *testing.T) {
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	python := []byte("# Source metadata reuse fixture\n")
	if err = os.WriteFile(filepath.Join(root, "pipeline_moss_image.py"), python, 0600); err != nil {
		t.Fatal(err)
	}
	pythonID, err := artifact.IdentifyBytes(artifact.KindFile, python)
	if err != nil {
		t.Fatal(err)
	}
	prior := artifact.Content{Descriptor: artifact.Descriptor{ID: pythonID, Size: uint64(len(python)), MediaType: "text/plain"}, Data: python}
	if _, err = store.Commit(t.Context(), artifact.Batch{Key: "native-source", Contents: []artifact.Content{prior}}); err != nil {
		t.Fatal(err)
	}
	before, _ := store.Head()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Register(ctx, store, root, ""); !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled registration", err)
	}
	if after, _ := store.Head(); before != after {
		t.Fatal("cancelled registration changed store")
	}
	receipt, err := Register(t.Context(), store, root, "")
	if err != nil {
		t.Fatal(err)
	}
	id, ok := receipt["recipe"].(artifact.ID)
	if held, found, err := store.Artifact(t.Context(), pythonID); err != nil || !found || held != prior.Descriptor {
		t.Fatal("source descriptor changed", held, err)
	}
	if !ok || receipt["qualification"] != "candidate" {
		t.Fatal("no executable candidate", receipt)
	}
	status, found, err := modelrecipe.Status(t.Context(), store, id)
	if err != nil || !found || status != recipe.StatusCandidate {
		t.Fatal("registration activated or omitted candidate", status, err)
	}
	before, sequence := store.Head()
	repeated, err := Register(t.Context(), store, root, "")
	if err != nil || repeated["recipe"] != id || repeated["model"] != receipt["model"] {
		t.Fatal("registration identity changed", repeated, err)
	}
	if after, next := store.Head(); after != before || next != sequence {
		t.Fatal("repeat registration changed journal")
	}
	model, definition, err := mediacapability.Prepare(t.Context(), store, root, recipe.TaskImageGen, mediacapability.Catalog[recipe.TaskImageGen])
	if err != nil || model != receipt["model"] || definition.ID != id {
		t.Fatal("command and library disagree", err)
	}
	if after, next := store.Head(); after != before || next != sequence {
		t.Fatal("shared preparation duplicated facts")
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = modelrecipe.CompileCandidateExecution(t.Context(), store, program); err != nil {
		t.Fatal("registered candidate cannot bind execution", err)
	}
	path := filepath.Join(root, "scheduler", "scheduler_config.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err = json.Unmarshal(b, &config); err != nil {
		t.Fatal(err)
	}
	config["use_dynamic_shifting"] = true
	b, err = json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	before, sequence = store.Head()
	if _, err = Register(t.Context(), store, root, ""); err == nil {
		t.Fatal("unsupported source policy registered")
	}
	if after, next := store.Head(); after != before || next != sequence {
		t.Fatal("invalid source changed published candidate")
	}
}
