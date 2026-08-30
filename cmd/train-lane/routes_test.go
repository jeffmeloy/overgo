package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

func routeFixtureGGUF(t *testing.T, architecture string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), architecture+".gguf")
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, architecture),
	}
	tensors := []gguf.TensorData{testutil.GGUFTensorF32("token_embd.weight", []uint64{4, 4}, 1)}
	if err := gguf.WriteFileExclusive(path, metadata, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return path
}

func routeFixtureDirectory(t *testing.T, modelType string) string {
	t.Helper()
	directory := t.TempDir()
	config := `{"model_type": "` + modelType + `"}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	return directory
}

// TestTrainerRoute pins the routing contract: each trainable input maps
// to the trainer its own declaration names — a declared route from the
// committed catalog with recorded facts substituted, the dense workflow
// for dense representations without a declaration, and a named refusal
// when a declared route needs a recorded fact that does not resolve.
func TestTrainerRoute(t *testing.T) {
	catalog, err := loadRouteCatalog(filepath.Join("..", "..", "docs", "training_routes.json"))
	if err != nil {
		t.Fatalf("committed route catalog: %v", err)
	}
	for _, key := range []string{"gguf:qwen35", "dir:t2v", "file:pt", "dir:unified_mot"} {
		if _, declared := catalog.Routes[key]; !declared {
			t.Fatalf("committed catalog declares no %q route", key)
		}
	}

	dense := []string{"go", "run", "./cmd/train", "-model", "dense-input"}
	denseGGUF := routeFixtureGGUF(t, "qwen2")
	route, err := resolveRoute(catalog, denseGGUF, "store-root", "", dense)
	if err != nil || route.Name != "dense" {
		t.Fatalf("dense gguf route = (%+v, %v), want the dense workflow", route, err)
	}

	hybridGGUF := routeFixtureGGUF(t, "qwen35")
	route, err = resolveRoute(catalog, hybridGGUF, "store-root", "", dense)
	if err != nil || route.Name != "gguf:qwen35" {
		t.Fatalf("hybrid gguf route = (%+v, %v), want gguf:qwen35", route, err)
	}
	assertSubstituted(t, route, hybridGGUF, "store-root")

	videoDir := routeFixtureDirectory(t, "t2v")
	route, err = resolveRoute(catalog, videoDir, "store-root", "", dense)
	if err != nil || route.Name != "dir:t2v" {
		t.Fatalf("t2v route = (%+v, %v), want dir:t2v", route, err)
	}
	assertSubstituted(t, route, videoDir, "store-root")

	motDir := routeFixtureDirectory(t, "unified_mot")
	route, err = resolveRoute(catalog, motDir, "store-root", "", dense)
	if err != nil || route.Name != "dir:unified_mot" {
		t.Fatalf("unified_mot route = (%+v, %v), want dir:unified_mot", route, err)
	}

	denseDir := routeFixtureDirectory(t, "qwen2")
	route, err = resolveRoute(catalog, denseDir, "store-root", "", dense)
	if err != nil || route.Name != "dense" {
		t.Fatalf("dense directory route = (%+v, %v), want the dense workflow", route, err)
	}

	checkpoint := filepath.Join(t.TempDir(), "weights.pt")
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	route, err = resolveRoute(catalog, checkpoint, "store-root", videoDir, dense)
	if err != nil || route.Name != "file:pt" {
		t.Fatalf("checkpoint route = (%+v, %v), want file:pt", route, err)
	}
	assertSubstituted(t, route, checkpoint, "store-root")
	found := false
	for _, argument := range route.Argv {
		if argument == videoDir {
			found = true
		}
	}
	if !found {
		t.Fatalf("checkpoint route %v does not bind the t2v stack directory", route.Argv)
	}
	if _, err = resolveRoute(catalog, checkpoint, "store-root", "", dense); err == nil {
		t.Fatal("checkpoint route resolved without a recorded t2v stack directory")
	}
}

// assertSubstituted requires every placeholder resolved and the input
// and store facts present in the resolved invocation.
func assertSubstituted(t *testing.T, route trainerRoute, input, store string) {
	t.Helper()
	haveInput, haveStore := false, false
	for _, argument := range route.Argv {
		if argument == input {
			haveInput = true
		}
		if argument == store {
			haveStore = true
		}
		for _, placeholder := range []string{"{input}", "{store}", "{t2v-dir}"} {
			if argument == placeholder {
				t.Fatalf("route %s keeps unresolved placeholder %s", route.Name, placeholder)
			}
		}
	}
	if !haveInput || !haveStore {
		t.Fatalf("route %s argv %v does not bind input=%s store=%s", route.Name, route.Argv, input, store)
	}
}
