package main

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type modelRegistration struct {
	Directory        string                   `json:"directory"`
	Model            artifact.ID              `json:"model"`
	TensorInventory  artifact.ID              `json:"tensor_inventory"`
	SourceRepository string                   `json:"source_repository"`
	SourceCommit     string                   `json:"source_commit"`
	Components       []modelartifact.FileSpec `json:"components,omitempty"`
	License          struct {
		Path      string      `json:"path"`
		Artifact  artifact.ID `json:"artifact"`
		SPDX      string      `json:"spdx"`
		MediaType string      `json:"media_type"`
	} `json:"license"`
}

func TestModelRegistrationIsAtomicIdempotentAndInactive(t *testing.T) {
	root, specification := registrationFixture(t)
	declaration := filepath.Join(t.TempDir(), "registration.json")
	repository := t.TempDir()
	write := func(values []modelRegistration) {
		t.Helper()
		data, err := json.Marshal(values)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(declaration, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	args := []string{"-repo", repository, "-root", root, "-spec", declaration}
	write([]modelRegistration{specification})
	if err := registerModels(args); err != nil {
		t.Fatal(err)
	}
	inspect := func() artifact.CommitID {
		t.Helper()
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, found, err := store.Manifest(t.Context(), specification.Model); err != nil || !found {
			t.Fatalf("registered model missing: %v", err)
		}
		results, err := store.Query(t.Context(), overgodb.Query{Kind: artifact.KindRecipe, MaxResults: 1, Projection: overgodb.ProjectArtifacts})
		if err != nil || results.Matched != 0 {
			t.Fatalf("registration created recipes: %+v, %v", results, err)
		}
		results, err = store.Query(t.Context(), overgodb.Query{MaxResults: 32, Projection: overgodb.ProjectAliases})
		if err != nil || len(results.Aliases) != 0 {
			t.Fatalf("registration changed activation/routing aliases: %+v, %v", results, err)
		}
		head, _ := store.Head()
		return head
	}
	head := inspect()
	if err := registerModels(args); err != nil {
		t.Fatal(err)
	}
	if inspect() != head {
		t.Fatal("idempotent registration advanced the journal")
	}
	for _, tc := range []struct {
		name   string
		change func(*modelRegistration)
	}{
		{"wrong model", func(value *modelRegistration) { value.Model = testutil.ArtifactID(t, artifact.KindModel, "wrong") }},
		{"wrong tensors", func(value *modelRegistration) {
			value.TensorInventory = testutil.ArtifactID(t, artifact.KindTensorInventory, "wrong")
		}},
		{"wrong source", func(value *modelRegistration) { value.SourceCommit = strings.Repeat("0", len(value.SourceCommit)) }},
		{"wrong license", func(value *modelRegistration) {
			value.License.Artifact = testutil.ArtifactID(t, artifact.KindFile, "wrong")
		}},
		{"root escape", func(value *modelRegistration) { value.Directory = "../outside" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := specification
			tc.change(&changed)
			write([]modelRegistration{changed})
			if err := registerModels(args); err == nil {
				t.Fatal("accepted invalid registration")
			}
			if inspect() != head {
				t.Fatal("failed registration changed the journal")
			}
		})
	}
	wrong := specification
	wrong.Model = testutil.ArtifactID(t, artifact.KindModel, "second invalid model")
	write([]modelRegistration{specification, wrong})
	if err := registerModels(args); err == nil {
		t.Fatal("accepted a partially valid batch")
	}
	if inspect() != head {
		t.Fatal("partial batch was published")
	}
	if _, err := os.Stat(filepath.Join(repository, "objects")); !os.IsNotExist(err) {
		t.Fatalf("registration copied objects: %v", err)
	}
	if _, err := taskModelInventory("missing", recipe.Task("unsupported")); err == nil {
		t.Fatal("unknown task was inferred as a GGUF")
	}
	fromStatus, err := taskModelInventory(filepath.Join(root, specification.Directory), recipe.TaskVQA)
	if err != nil || fromStatus.Manifest.ID != specification.Model {
		t.Fatalf("shared task inventory differs: %v", err)
	}
}

func registrationFixture(t *testing.T) (string, modelRegistration) {
	t.Helper()
	root := t.TempDir()
	directory := filepath.Join(root, "fixture")
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	header := `{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	weights := make([]byte, 8)
	binary.LittleEndian.PutUint64(weights, uint64(len(header)))
	weights = append(weights, header...)
	weights = append(weights, 0, 0, 128, 63)
	license := []byte("# Fixture license\n")
	for name, data := range map[string][]byte{
		"config.json":       []byte(`{"model_type":"fixture"}`),
		"tokenizer.json":    []byte(`{"version":"1.0"}`),
		"model.safetensors": weights, "README.md": license,
	} {
		if err := os.WriteFile(filepath.Join(directory, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	git := func(arguments ...string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-C", directory}, arguments...)...)
		command.Env = gitauthority.ReaderEnvironment()
		out, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("fixture Git %v: %v: %s", arguments, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("add", ".")
	git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--no-gpg-sign", "-qm", "fixture")
	source := "https://example.invalid/model"
	git("remote", "add", "origin", source)
	inventory, err := modelartifact.FromHFPath(directory)
	if err != nil {
		t.Fatal(err)
	}
	value := modelRegistration{Directory: "fixture", Model: inventory.Manifest.ID, TensorInventory: inventory.TensorInventory.ID, SourceRepository: source, SourceCommit: git("rev-parse", "HEAD")}
	value.License.Path, value.License.SPDX, value.License.MediaType = "README.md", "LicenseRef-Fixture", "text/markdown"
	value.License.Artifact, err = artifact.IdentifyBytes(artifact.KindFile, license)
	if err != nil {
		t.Fatal(err)
	}
	return root, value
}
