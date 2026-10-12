package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestScratchModuleWorkflow proves the supported scratch-module developer
// workflow on an isolated fixture with the same parent/nested-module contract as
// tmp: a nested module that replaces the parent lets a scratch file in a row's
// folder import the parent's internal package and run without an overlay from
// either the parent root or the nested directory, and the scratch package stays
// outside the parent's production ./... package set. No probe executable,
// overlay wrapper, or second module mechanism is introduced.
func TestScratchModuleWorkflow(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()
	write := func(rel, content string) {
		t.Helper()
		full := filepath.Join(parent, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	const marker = "scratch-linked-internal"
	write("go.mod", "module fixtureparent\n\ngo 1.26\n")
	write("internal/probe/probe.go", "package probe\n\n// Marker proves the scratch file linked the parent's internal package.\nconst Marker = \""+marker+"\"\n")
	write("tmp/go.mod", "module fixtureparent/tmp\n\ngo 1.26\n\nrequire fixtureparent v0.0.0\n\nreplace fixtureparent => ..\n")
	write("tmp/work/scratch.go", "package main\n\nimport (\n\t\"fmt\"\n\n\t\"fixtureparent/internal/probe\"\n)\n\nfunc main() { fmt.Print(probe.Marker) }\n")

	for _, invocation := range []struct{ dir, file string }{
		{parent, "./tmp/work/scratch.go"},
		{filepath.Join(parent, "tmp"), "./work/scratch.go"},
	} {
		out, err := commandOutput(invocation.dir, "go", "run", invocation.file)
		if err != nil || !strings.Contains(string(out), marker) {
			t.Fatalf("scratch run in %s: %v\n%s", invocation.dir, err, out)
		}
	}

	listed, err := commandOutput(parent, "go", "list", "./...")
	if err != nil {
		t.Fatalf("go list ./...: %v\n%s", err, listed)
	}
	if strings.Contains(string(listed), "fixtureparent/tmp") {
		t.Fatalf("scratch module leaked into the production package set:\n%s", listed)
	}
}

// TestPromptContractProjection holds that the prompt renders the bounded
// actionable contract -- title, verifier, prerequisites, commit scope, the
// row's scratch folder, a usable self-describing cmd/finding invocation, and
// the stop conditions -- while the rationale history stays out of the task
// prompt, available by reference.
func TestPromptContractProjection(t *testing.T) {
	t.Parallel()
	item := plan.Item{ID: "work", Title: "Work item"}
	step := plan.Step{
		ID: "do", Title: "Bounded change", Status: plan.StatusOpen, Verify: "go test ./x -run '^TestX$'",
		DependsOn: []string{"base/do"}, Rationale: strings.Repeat("buried history prose ", 20),
	}
	var out bytes.Buffer
	printPromptStep(item, step, &out)
	text := out.String()
	for _, want := range []string{"TASK work/do", "VERIFY go test", "PREREQUISITES base/do", "COMMIT go run ./cmd/loop -land work/do ", "SCRATCH tmp/work/ ", "go run ./cmd/finding -h", "STOP ", "Landed-Row"} {
		if !strings.Contains(text, want) {
			t.Fatalf("contract prompt missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "buried history prose") {
		t.Fatalf("rationale history leaked into the task prompt:\n%s", text)
	}
}

// TestPromptReadsWithoutWriting drives the real prompt path: it projects the
// dispatched task, and neither the store nor the plan changes, however many
// workers ask.
func TestPromptReadsWithoutWriting(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	root := initializePlanTestRepository(t, mutationPlan(t, "Prompt task"))
	t.Chdir(root)
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	for _, worker := range []string{"session-one", "session-two"} {
		out := captureStdout(t, func() {
			if err := run(cli{prompt: true, worker: worker}, nil); err != nil {
				t.Fatal(err)
			}
		})
		if !strings.Contains(out, "TASK row/do") {
			t.Fatalf("prompt for %s did not project the task:\n%s", worker, out)
		}
	}
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if after, count := store.Head(); after != head || count != sequence {
		t.Fatal("prompt wrote to the store")
	}
	if after, err := os.ReadFile(filepath.Join(root, plan.Path)); err != nil || !bytes.Equal(before, after) {
		t.Fatalf("prompt changed the plan: %v", err)
	}
}

func TestStopStatusProjection(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "stop-command-worker")
	t.Setenv(plan.AutomationModeEnvironment, plan.ExecutionInteractive)
	t.Setenv(plan.AutomationMaintenanceEnvironment, "")
	root := initializePlanTestRepository(t, mutationPlan(t, "stop projection"))
	t.Chdir(root)
	if _, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{}); err != nil {
		t.Fatal(err)
	}
	stopCLI := cli{stop: true, worker: "stop-command-worker"}
	var output bytes.Buffer
	if err := recordStopControlCommand(root, stopCLI, []string{"user-stop: operator paused"}, &output); err != nil {
		t.Fatal(err)
	}
	if _, err := commandOutput(root, "git", "-c", "core.hooksPath=", "commit", "--allow-empty", "-m", "another worker committed"); err != nil {
		t.Fatal(err)
	}
	var status plan.Dispatch
	rendered := captureStdout(t, func() {
		if err := run(cli{next: true, json: true}, nil); err != nil {
			t.Fatal(err)
		}
	})
	if err := json.Unmarshal([]byte(rendered), &status); err != nil {
		t.Fatal(err)
	}
	if status.Stop == nil || !status.Stop.Blocked || status.Stop.Event.CodeCommit == status.Stop.CurrentHead || status.Complete || status.Item != "" {
		t.Fatalf("stop projection=%+v", status)
	}
	prompt := captureStdout(t, func() {
		if err := run(cli{prompt: true, worker: stopCLI.worker}, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(prompt, "stop=active") || strings.Contains(prompt, "TASK ") {
		t.Fatalf("stopped executable prompt: %s", prompt)
	}
	raw, err := os.ReadFile(plan.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(plan.Path, []byte("{broken"), 0600); err != nil {
		t.Fatal(err)
	}
	captureStdout(t, func() {
		if err := run(cli{next: true}, nil); err != nil {
			t.Fatalf("stop status depended on live plan: %v", err)
		}
	})
	resume := cli{resumeStop: status.Stop.ID.String(), worker: stopCLI.worker}
	if err := recordStopControlCommand(root, resume, []string{"operator explicitly continued"}, &output); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(plan.Path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	resumed, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{})
	if err != nil || resumed.Stop == nil || resumed.Stop.State != "resumed" || resumed.Waiting != "" {
		t.Fatalf("resumed dispatch=%+v %v", resumed, err)
	}
}
