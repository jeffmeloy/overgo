package main

import (
	"crypto/sha1"
	"os"
	"overgo/internal/overgodb"
	"overgo/internal/worklease"
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

func TestPersistentLoopStop(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	writer, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	reader, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	world := &execWorld{stopStore: reader}
	if world.Paused() {
		t.Fatal("absent stop paused loop")
	}
	event := plan.ControlEvent{Kind: "stop", Lane: worklease.UnassignedRole, Worker: "loop-stop-worker", Worktree: filepath.ToSlash(root), Mode: plan.ExecutionInteractive, ReasonCode: "user-stop", Detail: "pause interactive session", CodeCommit: strings.Repeat("a", sha1.Size*2)}
	stopped, err := plan.RecordControlEvent(t.Context(), writer, event)
	if err != nil {
		t.Fatal(err)
	}
	if world.Paused() {
		t.Fatal("interactive scope stopped unattended work")
	}
	event.Previous = stopped.ID
	event.Mode = plan.ExecutionAll
	stopped, err = plan.RecordControlEvent(t.Context(), writer, event)
	if err != nil {
		t.Fatal(err)
	}
	if !world.Paused() {
		t.Fatal("existing loop reader missed new operator stop")
	}
	outcome, err := loop.Run(world, loop.Config{MaxInvocations: 1, MaxAttemptsPerStep: 1})
	if err != nil || outcome.Reason != loop.ReasonPaused || outcome.Invocations != 0 {
		t.Fatalf("stopped loop invoked work: %+v %v", outcome, err)
	}
	event.Kind = "resume"
	event.ReasonCode = "operator-resume"
	event.Detail = "operator continued"
	event.Previous = stopped.ID
	if _, err := plan.RecordControlEvent(t.Context(), writer, event); err != nil {
		t.Fatal(err)
	}
	if world.Paused() {
		t.Fatal("loop did not observe scoped resume")
	}
	t.Setenv(plan.AutomationModeEnvironment, plan.ExecutionInteractive)
	environment := loopEnvironment()
	last := ""
	for _, value := range environment {
		if strings.HasPrefix(value, plan.AutomationModeEnvironment+"=") {
			last = value
		}
	}
	if last != plan.AutomationModeEnvironment+"="+plan.ExecutionUnattended {
		t.Fatal("worker can inherit interactive scope")
	}
	if err := os.MkdirAll("docs", 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("docs/plan_stop.json", []byte("{invalid"), 0600); err != nil {
		t.Fatal(err)
	}
	if !world.Paused() {
		t.Fatal("contradictory legacy marker did not pause loop")
	}
}
