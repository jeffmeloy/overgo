package plan

import (
	"overgo/internal/worklease"
	"strings"
	"testing"
)

func frontierTestPlan() Plan {
	return Plan{Campaign: "frontier", Doctrine: "frontier doctrine", Items: []Item{
		{ID: "alpha", Title: "alpha", Status: StatusOpen, Steps: []Step{
			{ID: "one", Title: "one", Status: StatusOpen, Verify: "go test ./internal/plan"},
			{ID: "two", Title: "two", Status: StatusOpen, Verify: "go test ./internal/plan",
				DependsOn: []string{"alpha/one"}},
		}},
		{ID: "beta", Title: "beta", Status: StatusOpen, Steps: []Step{
			{ID: "solo", Title: "solo", Status: StatusOpen, Verify: "go test ./internal/plan"},
			{ID: "after", Title: "after", Status: StatusOpen, Verify: "go test ./internal/plan",
				DependsOn: []string{"gamma/pruned"}},
		}},
	}}
}

func frontierLease(t *testing.T, task, worktree, role string, claims worklease.WorkspaceClaims) worklease.Lease {
	t.Helper()
	lease, err := worklease.New(worklease.Lease{
		Task: task, Worktree: worktree, Branch: "lane/" + role, Role: role,
		TargetHead: strings.Repeat("a", 40), ConflictsWith: []string{},
		Resources: worklease.Resources{CPUThreads: 2, HostRAMGiB: 4},
		Claims:    claims, ExpiresAt: "2026-09-01T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease
}

// TestReadyFrontierLeaseIsolation pins the complete ordered ready frontier
// and its lease admission: dependencies gate frontier membership through the
// completion authority, Current stays a member of the frontier, and leases
// must own distinct frontier rows and roles with non-overlapping
// repo-relative claims whatever worktree each lease runs in.
func TestReadyFrontierLeaseIsolation(t *testing.T) {
	document := frontierTestPlan()
	if _, err := ReadyFrontier(document, CompletionAuthority{}); err == nil {
		t.Fatal("frontier without a resolved completion authority was served")
	}
	authority := testCompletionAuthority(t, document)
	frontier, err := ReadyFrontier(document, authority)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontier) != 2 || frontier[0].String() != "alpha/one" || frontier[1].String() != "beta/solo" {
		t.Fatalf("ready frontier = %v", frontier)
	}
	item, step, ok := Current(document, "", authority)
	if !ok || item.ID+"/"+step.ID != frontier[0].String() {
		t.Fatalf("Current %s/%s is not the first frontier row", item.ID, step.ID)
	}
	proven := testCompletionAuthority(t, document, "gamma/pruned")
	frontier, err = ReadyFrontier(document, proven)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontier) != 3 || frontier[2].String() != "beta/after" {
		t.Fatalf("evidence-widened frontier = %v", frontier)
	}

	alphaLease := frontierLease(t, "alpha/one", "lane-alpha", "builder",
		worklease.WorkspaceClaims{Write: []string{"internal/plan"}})
	betaLease := frontierLease(t, "beta/solo", "lane-beta", "verifier",
		worklease.WorkspaceClaims{Write: []string{"internal/overgodb"}})
	if err := ValidateFrontierLeases(frontier, []worklease.Lease{alphaLease, betaLease}); err != nil {
		t.Fatalf("isolated leases refused: %v", err)
	}
	offFrontier := frontierLease(t, "alpha/two", "lane-early", "eager",
		worklease.WorkspaceClaims{Write: []string{"cmd/plan"}})
	if err := ValidateFrontierLeases(frontier[:2], []worklease.Lease{offFrontier}); err == nil ||
		!strings.Contains(err.Error(), "outside the ready frontier") {
		t.Fatalf("off-frontier lease admitted: %v", err)
	}
	duplicate := frontierLease(t, "alpha/one", "lane-dup", "shadow",
		worklease.WorkspaceClaims{Write: []string{"cmd/gate"}})
	if err := ValidateFrontierLeases(frontier, []worklease.Lease{alphaLease, duplicate}); err == nil ||
		!strings.Contains(err.Error(), "both own frontier row") {
		t.Fatalf("duplicate row ownership admitted: %v", err)
	}
	sameRole := frontierLease(t, "beta/solo", "lane-role", "builder",
		worklease.WorkspaceClaims{Write: []string{"internal/loop"}})
	if err := ValidateFrontierLeases(frontier, []worklease.Lease{alphaLease, sameRole}); err == nil ||
		!strings.Contains(err.Error(), "one role owns one row at a time") {
		t.Fatalf("duplicate role admitted: %v", err)
	}
	overlapping := frontierLease(t, "beta/solo", "lane-clash", "verifier",
		worklease.WorkspaceClaims{Write: []string{"internal/plan/sync.go"}})
	if err := ValidateFrontierLeases(frontier, []worklease.Lease{alphaLease, overlapping}); err == nil ||
		!strings.Contains(err.Error(), "overlapping workspace claims") {
		t.Fatalf("cross-worktree claim overlap admitted: %v", err)
	}
	whole := frontierLease(t, "beta/solo", "lane-whole", "verifier", worklease.WorkspaceClaims{WholeWorktree: true})
	if err := ValidateFrontierLeases(frontier, []worklease.Lease{alphaLease, whole}); err == nil ||
		!strings.Contains(err.Error(), "overlapping workspace claims") {
		t.Fatalf("whole-worktree overlap admitted: %v", err)
	}
}

// TestMergeAcceptsGatedPrunedCompletion pins evidence-gated pruning: a merge
// accepts a row one side pruned only when the union of the parents'
// completion authorities proves that identity landed through the gate, and
// refuses evidence-less pruning, pruning beside concurrent edits, and item
// pruning with an unproven step.
func TestMergeAcceptsGatedPrunedCompletion(t *testing.T) {
	base := frontierTestPlan()
	local := frontierTestPlan()
	upstream := frontierTestPlan()
	upstream.Items[0].Steps = upstream.Items[0].Steps[1:] // upstream pruned alpha/one

	empty := testCompletionAuthority(t, base)
	if _, err := MergeDocumentsWithCompletion(base, local, upstream, empty, empty); err == nil ||
		!strings.Contains(err.Error(), "without gated completion evidence") {
		t.Fatalf("evidence-less step pruning accepted: %v", err)
	}
	proven := testCompletionAuthority(t, base, "alpha/one")
	merged, err := MergeDocumentsWithCompletion(base, local, upstream, empty, proven)
	if err != nil {
		t.Fatal(err)
	}
	if len(merged.Items[0].Steps) != 1 || merged.Items[0].Steps[0].ID != "two" {
		t.Fatalf("gated pruning result = %+v", merged.Items[0])
	}
	if _, err := MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("authority-free MergeDocuments accepted a pruning")
	}

	edited := frontierTestPlan()
	edited.Items[0].Steps[0].Rationale = "concurrent local edit"
	if _, err := MergeDocumentsWithCompletion(base, edited, upstream, empty, proven); err == nil ||
		!strings.Contains(err.Error(), "beside concurrent edits") {
		t.Fatalf("pruning beside concurrent edits accepted: %v", err)
	}

	prunedItem := frontierTestPlan()
	prunedItem.Items = prunedItem.Items[1:] // upstream pruned the whole alpha item
	if _, err := MergeDocumentsWithCompletion(base, local, prunedItem, empty, proven); err == nil ||
		!strings.Contains(err.Error(), "lacks gated completion evidence") {
		t.Fatalf("item pruning with an unproven step accepted: %v", err)
	}
	fullyProven := testCompletionAuthority(t, base, "alpha/one", "alpha/two")
	merged, err = MergeDocumentsWithCompletion(base, local, prunedItem, empty, fullyProven)
	if err != nil {
		t.Fatal(err)
	}
	if len(merged.Items) != 1 || merged.Items[0].ID != "beta" {
		t.Fatalf("gated item pruning result = %+v", merged.Items)
	}
}

// TestMergePreservesDependencyOrder pins that an accepted pruning keeps the
// dependency graph intact: retained steps keep their references to the pruned
// row, document order survives the merge, and the merged frontier still
// blocks the dependent until the completion authority proves the pruned
// producer.
func TestMergePreservesDependencyOrder(t *testing.T) {
	base := frontierTestPlan()
	local := frontierTestPlan()
	upstream := frontierTestPlan()
	upstream.Items[0].Steps = upstream.Items[0].Steps[1:]

	empty := testCompletionAuthority(t, base)
	proven := testCompletionAuthority(t, base, "alpha/one")
	merged, err := MergeDocumentsWithCompletion(base, local, upstream, empty, proven)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Items[0].ID != "alpha" || merged.Items[1].ID != "beta" {
		t.Fatalf("merge reordered items: %+v", merged.Items)
	}
	dependent := merged.Items[0].Steps[0]
	if dependent.ID != "two" || len(dependent.DependsOn) != 1 || dependent.DependsOn[0] != "alpha/one" {
		t.Fatalf("merge lost the dependency on the pruned row: %+v", dependent)
	}
	unproven := testCompletionAuthority(t, merged)
	frontier, err := ReadyFrontier(merged, unproven)
	if err != nil {
		t.Fatal(err)
	}
	for _, ref := range frontier {
		if ref.String() == "alpha/two" {
			t.Fatalf("dependent row dispatched without completion evidence: %v", frontier)
		}
	}
	sequenced := testCompletionAuthority(t, merged, "alpha/one")
	frontier, err = ReadyFrontier(merged, sequenced)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontier) == 0 || frontier[0].String() != "alpha/two" {
		t.Fatalf("proven producer did not unblock the dependent in order: %v", frontier)
	}
}
