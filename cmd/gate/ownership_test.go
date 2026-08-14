package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gatecontrol"
)

func TestNonGoOwnershipGateScope(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: repo, paths: []string{"internal/model/architecture_profiles.json"}}
	owners, err := g.directChangedPackages()
	if err != nil {
		t.Fatal(err)
	}
	if !owners["overgo/internal/model"] {
		t.Fatalf("embedded architecture catalog owner missing: %v", owners)
	}
	if _, err := os.Stat(filepath.Join(repo, "internal", "model", "architecture_profiles.json")); err != nil {
		t.Fatal(err)
	}
}

func TestAutomationCommonOwners(t *testing.T) {
	control, err := gatecontrol.New(t.TempDir(), "store")
	if err != nil {
		t.Fatal(err)
	}
	cache := control.Retry("tree", "env-a")
	cache.MarkSucceeded("test")
	if err := control.SaveRetry(cache); err != nil {
		t.Fatal(err)
	}
	if !control.Retry("tree", "env-a").Succeeded("test") {
		t.Fatal("matching environment did not reuse cache")
	}
	if control.Retry("tree", "env-b").Succeeded("test") {
		t.Fatal("retry cache crossed environment identity")
	}
}
