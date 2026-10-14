package worklease

import (
	"slices"
	"testing"
	"time"
)

const orchestrationHead = "0123456789abcdef0123456789abcdef01234567"

// leaseFixture returns one advisory lease of a task on a worktree, as a plain value.
//   - Claims = the whole worktree, which is what a lease that names no path claims
func leaseFixture(task, worktree string) Lease {
	return Lease{
		Version: 1, Task: task, Worktree: worktree, Branch: "codex/" + task, Role: "developer", TargetHead: orchestrationHead,
		ConflictsWith: []string{}, Resources: Resources{CPUThreads: 8, HostRAMGiB: 16, VRAMGiB: 12}, ExpiresAt: "2026-08-15T00:00:00Z",
		Claims: WorkspaceClaims{WholeWorktree: true},
	}
}

func TestResourceAdvisory(t *testing.T) {
	first := leaseFixture("first/do", "C:/repo/first")
	first.Resources.GPUExclusive = true
	second := leaseFixture("second/do", "C:/repo/second")
	advisory := AssessResources(time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC), Resources{CPUThreads: 16, HostRAMGiB: 96, VRAMGiB: 48}, []Lease{second, first})
	if advisory.Fits || advisory.Reserved.CPUThreads != 16 || !slices.Equal(advisory.ActiveTasks, []string{"first/do", "second/do"}) || len(advisory.Conflicts) != 1 {
		t.Fatalf("resource advisory = %+v", advisory)
	}
}
