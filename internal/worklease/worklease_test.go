package worklease

import (
	"slices"
	"testing"
	"time"
)

const orchestrationHead = "0123456789abcdef0123456789abcdef01234567"

func leaseFixture(t *testing.T, task, worktree string) Lease {
	t.Helper()
	lease, err := Codec.New(Lease{
		Version: Version, Task: task, Worktree: worktree, Branch: "codex/" + task, Role: "developer", TargetHead: orchestrationHead,
		ConflictsWith: []string{}, Resources: Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 12}, ExpiresAt: "2026-08-15T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease
}

func TestWorkLease(t *testing.T) {
	lease := leaseFixture(t, "automation/do", "C:/repo/automation")
	content, err := Codec.Content(lease)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Codec.Parse(content.Data)
	if err != nil || parsed.ID != lease.ID || worktreeAlias(parsed.Worktree) == "" {
		t.Fatalf("work lease round trip = (%+v, %v)", parsed, err)
	}
}

func TestResourceAdvisory(t *testing.T) {
	first := leaseFixture(t, "first/do", "C:/repo/first")
	first.Resources.GPUExclusive = true
	first, _ = Codec.New(first)
	second := leaseFixture(t, "second/do", "C:/repo/second")
	advisory := AssessResources(time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC), Resources{CPUThreads: 16, HostRAMGiB: 96, VRAMGiB: 48}, []Lease{second, first})
	if advisory.Fits || advisory.Reserved.CPUThreads != 16 || !slices.Equal(advisory.ActiveTasks, []string{"first/do", "second/do"}) || len(advisory.Conflicts) != 1 {
		t.Fatalf("resource advisory = %+v", advisory)
	}
}
