package plan

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const orchestrationHead = "0123456789abcdef0123456789abcdef01234567"

func workLeaseFixture(t *testing.T, task, worktree string) WorkLease {
	t.Helper()
	lease, err := NewWorkLease(WorkLease{
		Task: task, Worktree: worktree, Branch: "codex/" + task, Role: "developer", TargetHead: orchestrationHead,
		DependsOn: []string{}, ConflictsWith: []string{}, Resources: ResourceRequest{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 12},
		EvidenceLanes: []string{"go-test"}, ExpiresAt: "2026-08-15T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease
}

func TestWorkLease(t *testing.T) {
	lease := workLeaseFixture(t, "automation/do", "C:/repo/automation")
	content, err := lease.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseWorkLease(content.Data)
	if err != nil || parsed.ID != lease.ID || WorkLeaseAlias(parsed.Worktree) == "" {
		t.Fatalf("work lease round trip = (%+v, %v)", parsed, err)
	}
	batch, err := WorkLeaseBatch(lease, nil)
	if err != nil || len(batch.Aliases) != 1 || batch.Aliases[0].Target != lease.ID {
		t.Fatalf("work lease batch = (%+v, %v)", batch, err)
	}
}

func TestResourceAdvisory(t *testing.T) {
	first := workLeaseFixture(t, "first/do", "C:/repo/first")
	first.Resources.GPUExclusive = true
	first, _ = NewWorkLease(first)
	second := workLeaseFixture(t, "second/do", "C:/repo/second")
	advisory := AssessResources(time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC), ResourceCapacity{CPUThreads: 16, HostRAMGiB: 96, VRAMGiB: 48}, []WorkLease{second, first})
	if advisory.Fits || advisory.Reserved.CPUThreads != 16 || !slices.Equal(advisory.ActiveTasks, []string{"first/do", "second/do"}) || len(advisory.Conflicts) != 1 {
		t.Fatalf("resource advisory = %+v", advisory)
	}
}

func TestMergeEligibility(t *testing.T) {
	lease := workLeaseFixture(t, "candidate/do", "C:/repo/candidate")
	candidate := "89abcdef0123456789abcdef0123456789abcdef"
	verdict := testutil.ArtifactID(t, artifact.KindEvidence, "approved verdict")
	required := testutil.ArtifactID(t, artifact.KindEvidence, "gate result")
	input := MergeEligibilityInput{
		CurrentTargetHead: orchestrationHead, CandidateHead: candidate, Lease: lease, ActiveLease: lease.ID,
		WorktreeClean: true, ReviewVerdict: verdict, RequiredEvidence: []artifact.ID{required}, ObservedEvidence: []artifact.ID{required},
	}
	if got := AssessMergeEligibility(input); !got.Eligible || !got.OwnerDecisionRequired || len(got.Reasons) != 0 {
		t.Fatalf("eligible packet = %+v", got)
	}
	input.CurrentTargetHead = candidate
	if got := AssessMergeEligibility(input); got.Eligible || len(got.Reasons) == 0 {
		t.Fatalf("moved target admitted = %+v", got)
	}
	input.CurrentTargetHead = orchestrationHead
	input.Lease.Role = "sqa"
	if got := AssessMergeEligibility(input); got.Eligible || !slices.Contains(got.Reasons, "worktree lease identity is invalid") {
		t.Fatalf("mutated lease admitted = %+v", got)
	}
}
