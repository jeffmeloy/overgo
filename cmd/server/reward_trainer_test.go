package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingworkflow"
)

// TestRewardTrainerIsFoundOrRefused holds how a server finds the training
// command a reward run from the workbench executes in.
//
// With no file at the path it is given, the server has no way to run one, and
// says so by returning none: the training workspace then lists reward
// training with the reason it cannot run, and the server still starts.
//
// With a file there, what it returns runs a request in that command, as a
// child process. A request held to the host, which the command's flags cannot
// carry, comes back with that route's own refusal and nothing is started: the
// file here is no program, and the refusal does not name it. A request the
// flags can carry is started as the file at that path, and the failed start
// names it.
func TestRewardTrainerIsFoundOrRefused(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	if run := rewardTrainer(filepath.Join(directory, "absent.exe"), filepath.Join(directory, "store")); run != nil {
		t.Fatal("a path that names no file gave a way to run a reward session; want none, so that the workspace lists the refusal")
	}
	command := filepath.Join(directory, "overgo-train.exe")
	if err := os.WriteFile(command, []byte("no program"), 0o600); err != nil {
		t.Fatal(err)
	}
	run := rewardTrainer(command, filepath.Join(directory, "store"))
	if run == nil {
		t.Fatal("a path that names a file gave no way to run a reward session")
	}
	store, err := overgodb.Open(filepath.Join(directory, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	request := trainingworkflow.Request{
		Repository: store, Observations: store, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "reward trainer recipe"),
		ModelDirectory: filepath.Join(directory, "model"), DatasetPath: filepath.Join(directory, "rollouts.jsonl"),
		OutputDirectory: filepath.Join(directory, "trained"), Steps: 1,
	}
	held := request
	held.Host = true
	if _, err := run(t.Context(), held); err == nil || !strings.Contains(err.Error(), "this request asks for something else") || strings.Contains(err.Error(), command) {
		t.Fatalf("a request held to the host = %v; want the training command's route to refuse it before it starts %s", err, command)
	}
	if _, err := run(t.Context(), request); err == nil || !strings.Contains(err.Error(), "start "+command) {
		t.Fatalf("a request the training command can carry = %v; want it started as %s, which is no program", err, command)
	}
}
