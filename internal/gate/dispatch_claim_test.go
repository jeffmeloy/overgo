package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func TestClaimedGateAdmission(t *testing.T) {
	t.Setenv(plan.AutomationRoleEnvironment, plan.UnassignedRole)
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
	if err := plan.ResolveWorkLeaseOwner(t.Context(), store, *claimed.Claim); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, release); err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Artifact(t.Context(), marker); err != nil || !found {
		t.Fatalf("completion marker missing: %v", err)
	}
	if err := plan.ResolveWorkLeaseOwner(t.Context(), store, *claimed.Claim); err == nil {
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
