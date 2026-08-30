package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/testutil"
)

func TestBlockSharesPlanGateMutationLock(t *testing.T) {
	root := t.TempDir()
	census := testutil.ArtifactID(t, artifact.KindEvidence, "loop block census")
	document := plan.Plan{Census: &census, Items: []plan.Item{{
		ID: "row", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/loop"}},
	}}}
	path := filepath.Join(root, filepath.FromSlash(plan.Path))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lock, err := authoritylock.Acquire(root)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	t.Chdir(root)
	err = (&execWorld{}).Block(loop.Step{Item: "row", ID: "do"}, "fixture")
	if err == nil || !strings.Contains(err.Error(), "mutation is active") {
		t.Fatalf("block under gate lock = %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("refused block changed plan bytes")
	}
}
