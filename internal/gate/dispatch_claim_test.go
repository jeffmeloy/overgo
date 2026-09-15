package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/worklease"
)

func TestClaimedGateAdmission(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "claim-worker")
	document := plan.Plan{Items: []plan.Item{{ID: "row", Status: plan.StatusOpen, Steps: []plan.Step{
		{ID: "first", Status: plan.StatusOpen, Verify: "go test ./internal/plan"},
		{ID: "independent", Status: plan.StatusOpen, Verify: "go test ./internal/gate"},
	}}}}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "docs"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(root, plan.Path), document); err != nil {
		t.Fatal(err)
	}
	initializePlanBindingRepo(t, root)
	t.Setenv(plan.AutomationWorkerEnvironment, "claim-worker")
	dirty := []byte("package pending // preserve uncommitted work\n")
	if err := os.WriteFile(filepath.Join(root, "pending.go"), dirty, 0o600); err != nil {
		t.Fatal(err)
	}
	claimed, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Acquire: true, Reference: "row/independent"})
	if err != nil || claimed.Claim == nil || claimed.ClaimID != claimed.Claim.ID {
		t.Fatalf("executable dispatch=%+v %v", claimed, err)
	}
	authority, _, admitted, err := resolvePlanBinding(root, gateStorePath, "row/independent")
	if err != nil || admitted == nil || admitted.ID != claimed.ClaimID {
		t.Fatalf("claim handoff=%+v %v", admitted, err)
	}
	if _, err := completionAcceptanceEvidenceForPlan(document, "row/independent", authority); err != nil {
		t.Fatalf("claimed ready acceptance: %v", err)
	}
	if err := checkPlanBindingForTest(root, "row/first"); err == nil {
		t.Fatal("gate admitted a row this worker did not claim")
	}
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	if err := checkPlanBindingForTest(root, "row/independent"); err == nil {
		t.Fatal("unassigned legacy caller bypassed active ownership")
	}
	t.Setenv(plan.AutomationWorkerEnvironment, "claim-worker")
	document.Items[0].Steps[1].Verify = "go test ./internal/artifact"
	if err := plan.Save(filepath.Join(root, plan.Path), document); err != nil {
		t.Fatal(err)
	}
	if err := checkPlanBindingForTest(root, "row/independent"); err == nil {
		t.Fatal("contract changed after claim but final admission accepted it")
	}
	document.Items[0].Steps[1].Verify = "go test ./internal/gate"
	if err := plan.Save(filepath.Join(root, plan.Path), document); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// The finalization owner composes release with its evidence batch. A stale
	// alias precondition rejects the entire batch, preserving the active claim.
	release, err := claimed.Claim.ReleaseBatch("claim-worker", "completed")
	if err != nil {
		t.Fatal(err)
	}
	marker, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("completion recorded atomically with claim release"))
	if err != nil {
		t.Fatal(err)
	}
	release.Artifacts = append(release.Artifacts, artifact.Descriptor{ID: marker})
	invalid := release
	invalid.Aliases = slices.Clone(release.Aliases)
	invalid.Aliases[0].Previous = &marker
	if _, err := artifact.CommitBatch(t.Context(), store, invalid); err == nil {
		t.Fatal("stale release admitted")
	}
	if _, found, err := store.Artifact(t.Context(), marker); err != nil || found {
		t.Fatalf("partial completion escaped rejected release: %v", err)
	}
	if err := worklease.ResolveOwner(t.Context(), store, *claimed.Claim); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, release); err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Artifact(t.Context(), marker); err != nil || !found {
		t.Fatalf("completion marker missing: %v", err)
	}
	if err := worklease.ResolveOwner(t.Context(), store, *claimed.Claim); err == nil {
		t.Fatal("successful completion retained claim")
	}
	if err := checkPlanBindingForTest(root, "row/independent"); err == nil {
		t.Fatal("released claim retained final admission")
	}
	actual, err := os.ReadFile(filepath.Join(root, "pending.go"))
	if err != nil || !bytes.Equal(actual, dirty) {
		t.Fatal("claim lifecycle overwrote dirty source")
	}
}

func TestPersistentGateStop(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "docs"), 0700); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Items: []plan.Item{{ID: "row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./internal/gate"}}}}}
	if err := plan.Save(filepath.Join(root, plan.Path), document); err != nil {
		t.Fatal(err)
	}
	initializePlanBindingRepo(t, root)
	t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
	t.Setenv(plan.AutomationWorkerEnvironment, "stop-gate-worker")
	t.Setenv(plan.AutomationModeEnvironment, plan.ExecutionInteractive)
	t.Setenv(plan.AutomationMaintenanceEnvironment, "")
	head, err := command(root, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	head = strings.TrimSpace(head)
	store, err := overgodb.Open(filepath.Join(root, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	event := plan.ControlEvent{Kind: "stop", Lane: worklease.UnassignedRole, Worker: "stop-gate-worker", Worktree: filepath.ToSlash(root), Mode: plan.ExecutionAll, ReasonCode: "user-stop", Detail: "operator stopped autonomous work", CodeCommit: head}
	stopped, err := plan.RecordControlEvent(t.Context(), store, event)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkPlanBindingForTest(root, "row/do"); err == nil {
		t.Fatal("gate ignored active stop")
	}
	grant := event
	grant.Kind = "maintenance"
	grant.ReasonCode = "operator-maintenance"
	grant.Detail = "operator authorized this repair"
	grant.Task = "row/do"
	grant.Previous = stopped.ID
	granted, err := plan.RecordControlEvent(t.Context(), store, grant)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(plan.AutomationMaintenanceEnvironment, granted.ID.String())
	claimed, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Acquire: true, Reference: "row/do"})
	if err != nil || claimed.Claim == nil || claimed.Waiting != "" {
		t.Fatalf("maintenance dispatch=%+v %v", claimed, err)
	}
	if err := checkPlanBindingForTest(root, "row/do"); err != nil {
		t.Fatal(err)
	}
	t.Setenv(plan.AutomationModeEnvironment, plan.ExecutionUnattended)
	if err := checkPlanBindingForTest(root, "row/do"); err == nil {
		t.Fatal("unattended gate consumed maintenance grant")
	}
	t.Setenv(plan.AutomationModeEnvironment, plan.ExecutionInteractive)
	event.Previous = stopped.ID
	event.Detail = "operator replaced the stop after admission"
	latest, err := plan.RecordControlEvent(t.Context(), store, event)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkPlanBindingForTest(root, "row/do"); err == nil {
		t.Fatal("final admission reused superseded stop authority")
	}
	active, err := plan.ReadStop(t.Context(), store, root, head, plan.ExecutionUnattended)
	if err != nil || !active.Blocked || active.ID != latest.ID {
		t.Fatalf("maintenance resumed loop: %+v %v", active, err)
	}
}
