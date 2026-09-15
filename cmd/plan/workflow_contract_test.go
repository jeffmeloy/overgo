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

func TestDispatchClaimProjection(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	document := mutationPlan(t, "claimed task")
	root := initializePlanTestRepository(t, document)
	t.Chdir(root)
	source := filepath.Join(root, "worker-edits.go")
	dirty := []byte("package pending // another worker's uncommitted source\n")
	if err := os.WriteFile(source, dirty, 0o600); err != nil {
		t.Fatal(err)
	}
	inspection, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{})
	if err != nil || inspection.Claim != nil || inspection.Item != "row" {
		t.Fatalf("inspection=%+v %v", inspection, err)
	}
	request := cli{prompt: true, worker: "session-one", retireLegacyLeases: noLegacyLeaseRetirement}
	output := captureStdout(t, func() {
		if err := run(request, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(output, "Claim:") || !strings.Contains(output, "worker=session-one") {
		t.Fatalf("prompt missing ownership: %s", output)
	}
	owned, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Worker: request.worker})
	if err != nil || owned.Claim == nil || owned.Waiting != "" {
		t.Fatalf("claimed inspection=%+v %v", owned, err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	captureStdout(t, func() {
		if err := run(request, nil); err != nil {
			t.Fatal(err)
		}
	})
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if after, count := store.Head(); after != head || count != sequence {
		t.Fatal("prompt retry duplicated claim")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	request.worker = "session-two"
	output = captureStdout(t, func() {
		if err := run(request, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(output, "plan waiting:") || strings.Contains(output, "Claim:") {
		t.Fatalf("competitor received executable task: %s", output)
	}
	before, err := os.ReadFile(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	err = withPlanMutation(root, false, func(current plan.Plan) error {
		current.Items[0].Steps[0].Title = "replanned beneath active source"
		return savePlanMutation(root, current)
	})
	if err == nil || !strings.Contains(err.Error(), "claimed by worker session-one") {
		t.Fatalf("claimed mutation=%v", err)
	}
	after, err := os.ReadFile(filepath.Join(root, plan.Path))
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("refused mutation changed plan")
	}
	var released bytes.Buffer
	if err := releaseDispatchClaim(root, owned.Claim.ID.String(), "session-two", "handoff", &released); err == nil {
		t.Fatal("foreign worker released ownership")
	}
	if err := releaseDispatchClaim(root, owned.Claim.ID.String(), "session-one", "completed", &released); err == nil {
		t.Fatal("CLI claimed completion without a gate")
	}
	if err := run(cli{releaseClaim: owned.Claim.ID.String(), releaseReason: "handoff", worker: "session-one", add: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err == nil {
		t.Fatal("mixed release and mutation options were admitted")
	}
	if err := releaseDispatchClaim(root, owned.Claim.ID.String(), "session-one", "handoff", &released); err != nil {
		t.Fatal(err)
	}
	if err := withPlanMutation(root, false, func(current plan.Plan) error {
		current.Items[0].Steps[0].Title = "reviewed after handoff"
		return savePlanMutation(root, current)
	}); err != nil {
		t.Fatal(err)
	}
	output = captureStdout(t, func() {
		if err := run(request, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(output, "worker=session-two") {
		t.Fatalf("handoff failed: %s", output)
	}
	afterSource, err := os.ReadFile(source)
	if err != nil || !bytes.Equal(afterSource, dirty) {
		t.Fatal("dispatch or plan amendment changed dirty source")
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
	stopCLI := cli{stop: true, worker: "stop-command-worker", retireLegacyLeases: noLegacyLeaseRetirement}
	var output bytes.Buffer
	if err := recordStopControlCommand(root, stopCLI, []string{"user-stop: operator paused"}, &output); err != nil {
		t.Fatal(err)
	}
	if _, err := commandOutput(root, "git", "-c", "core.hooksPath=", "commit", "--allow-empty", "-m", "another worker committed"); err != nil {
		t.Fatal(err)
	}
	var status plan.Dispatch
	rendered := captureStdout(t, func() {
		if err := run(cli{next: true, json: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	if err := json.Unmarshal([]byte(rendered), &status); err != nil {
		t.Fatal(err)
	}
	if status.Stop == nil || !status.Stop.Blocked || status.Stop.Event.CodeCommit == status.Stop.CurrentHead || status.Complete || status.Claim != nil {
		t.Fatalf("stop projection=%+v", status)
	}
	prompt := captureStdout(t, func() {
		if err := run(cli{prompt: true, worker: stopCLI.worker, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(prompt, "stop=active") || strings.Contains(prompt, "Claim:") {
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
		if err := run(cli{next: true, retireLegacyLeases: noLegacyLeaseRetirement}, nil); err != nil {
			t.Fatalf("stop status depended on live plan: %v", err)
		}
	})
	resume := cli{resumeStop: status.Stop.ID.String(), worker: stopCLI.worker, retireLegacyLeases: noLegacyLeaseRetirement}
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
