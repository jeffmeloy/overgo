package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelartifact"
)

func TestRecipeSpecPipelineRegistration(t *testing.T) {
	root, _ := registrationFixture(t)
	directory := filepath.Join(root, "fixture")
	component := filepath.Join(directory, "transformer")
	if err := os.MkdirAll(component, 0o700); err != nil {
		t.Fatal(err)
	}
	// Reuse the fixture's validated payload in the component declared by the pipeline.
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if filepath.Ext(name) != ".safetensors" && name != "config.json" {
			continue
		}
		body, err := os.ReadFile(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(component, name), body, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(directory, "model_index.json"), []byte(`{"_class_name":"MOSSImagePipeline","transformer":["pipeline_moss_image","MOSSImageTransformer"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(directory, "config.json")); err != nil {
		t.Fatal(err)
	}
	fixtureGit(t, directory, "add", ".")
	fixtureGit(t, directory, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--no-gpg-sign", "-qm", "pipeline")
	output := filepath.Join(t.TempDir(), "pipeline.json")
	if err := assembleSpecification([]string{"-root", root, "-output", output, "-spdx", "fixture=Apache-2.0", "fixture"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var got []modelRegistration
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	want, err := modelartifact.FromHFPath(directory)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Model != want.Manifest.ID || got[0].TensorInventory != want.TensorInventory.ID {
		t.Fatalf("declaration does not bind pipeline: %+v", got)
	}
	if err := registerModels([]string{"-repo", t.TempDir(), "-root", root, "-spec", output}); err != nil {
		t.Fatalf("register refused assembled pipeline: %v", err)
	}
}
