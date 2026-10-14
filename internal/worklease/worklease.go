// Package worklease owns work leases and dispatch claims: the lane
// assignments and the workspace claims that decide whether two leases
// collide. It sits below the plan so that operation runtimes consume leases
// without compiling the plan's readers.
package worklease

import (
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
)

// Resources is advisory capacity metadata; it never acquires hardware.
type Resources struct {
	CPUThreads   int  `json:"cpu_threads"`
	HostRAMGiB   int  `json:"host_ram_gib"`
	VRAMGiB      int  `json:"vram_gib"`
	GPUExclusive bool `json:"gpu_exclusive"`
}

// WorkspaceClaims declares exact path access. WholeWorktree is the
// conservative representation for an absent or unresolvable scope.
type WorkspaceClaims struct {
	WholeWorktree bool     `json:"whole_worktree,omitzero"`
	Read          []string `json:"read,omitempty"`
	Write         []string `json:"write,omitempty"`
}

// Lease is one lane assignment and what it reserves.
type Lease struct {
	Version       uint16          `json:"version"`
	Task          string          `json:"task"`
	Worktree      string          `json:"worktree"`
	Branch        string          `json:"branch"`
	Role          string          `json:"role"`
	TargetHead    string          `json:"target_head"`
	ConflictsWith []string        `json:"conflicts_with"`
	Resources     Resources       `json:"resources"`
	Claims        WorkspaceClaims `json:"claims"`
	ExpiresAt     string          `json:"expires_at"`
	// Dispatch claims retain their worker and contract until explicit release.
	// Acquisition binds the observed store head, distinguishing reacquisition.
	Worker      string      `json:"worker,omitzero"`
	Contract    string      `json:"contract,omitzero"`
	Acquisition string      `json:"acquisition,omitzero"`
	Previous    artifact.ID `json:"previous,omitzero"`
	// Optional experiment retry state.
	Experiment      artifact.ID `json:"experiment,omitzero"`
	Checkpoint      artifact.ID `json:"checkpoint,omitzero"`
	Retry           uint32      `json:"retry,omitzero"`
	PredictedWallNS uint64      `json:"predicted_wall_ns,omitzero"`
	ID              artifact.ID `json:"-"`
}

// WorkspaceClaimsConflict reports write/read or write/write overlap.
func WorkspaceClaimsConflict(left, right Lease) bool {
	if !strings.EqualFold(filepath.Clean(left.Worktree), filepath.Clean(right.Worktree)) {
		return false
	}
	if left.Claims.WholeWorktree || right.Claims.WholeWorktree {
		return true
	}
	for _, write := range left.Claims.Write {
		for _, path := range append(slices.Clone(right.Claims.Read), right.Claims.Write...) {
			if ClaimPathsOverlap(write, path) {
				return true
			}
		}
	}
	for _, write := range right.Claims.Write {
		for _, read := range left.Claims.Read {
			if ClaimPathsOverlap(write, read) {
				return true
			}
		}
	}
	return false
}

// ClaimPathsOverlap reports equal or nested claim paths, case-insensitively.
func ClaimPathsOverlap(left, right string) bool {
	left, right = strings.TrimSuffix(filepath.ToSlash(filepath.Clean(left)), "/"), strings.TrimSuffix(filepath.ToSlash(filepath.Clean(right)), "/")
	return strings.EqualFold(left, right) || strings.HasPrefix(strings.ToLower(left), strings.ToLower(right)+"/") || strings.HasPrefix(strings.ToLower(right), strings.ToLower(left)+"/")
}

// ResourceAdvisory reports active reservations and their collisions.
type ResourceAdvisory struct {
	ActiveTasks []string  `json:"active_tasks"`
	Reserved    Resources `json:"reserved"`
	Fits        bool      `json:"fits"`
	Conflicts   []string  `json:"conflicts"`
}

// AssessResources reports active reservations and collisions. Zero capacity
// means unknown, not zero available; this remains advice for the owner.
func AssessResources(now time.Time, capacity Resources, leases []Lease) ResourceAdvisory {
	active := make([]Lease, 0, len(leases))
	for _, lease := range leases {
		expires, err := time.Parse(time.RFC3339Nano, lease.ExpiresAt)
		if lease.Worker != "" || err == nil && expires.After(now) {
			active = append(active, lease)
		}
	}
	sort.Slice(active, func(i, j int) bool { return active[i].Task < active[j].Task })
	result := ResourceAdvisory{Fits: true, ActiveTasks: make([]string, 0, len(active)), Conflicts: []string{}}
	for _, lease := range active {
		result.ActiveTasks = append(result.ActiveTasks, lease.Task)
		result.Reserved.CPUThreads += lease.Resources.CPUThreads
		result.Reserved.HostRAMGiB += lease.Resources.HostRAMGiB
		result.Reserved.VRAMGiB += lease.Resources.VRAMGiB
		result.Reserved.GPUExclusive = result.Reserved.GPUExclusive || lease.Resources.GPUExclusive
	}
	for i := range active {
		for j := i + 1; j < len(active); j++ {
			left, right := active[i], active[j]
			switch {
			case WorkspaceClaimsConflict(left, right):
				result.Conflicts = append(result.Conflicts, fmt.Sprintf("%s and %s share worktree %s", left.Task, right.Task, left.Worktree))
			case slices.Contains(left.ConflictsWith, right.Task) || slices.Contains(right.ConflictsWith, left.Task):
				result.Conflicts = append(result.Conflicts, fmt.Sprintf("%s conflicts with %s", left.Task, right.Task))
			case (left.Resources.GPUExclusive || right.Resources.GPUExclusive) && left.Resources.VRAMGiB > 0 && right.Resources.VRAMGiB > 0:
				result.Conflicts = append(result.Conflicts, fmt.Sprintf("%s and %s both reserve the exclusive GPU", left.Task, right.Task))
			}
		}
	}
	for _, check := range []struct {
		name     string
		reserved int
		capacity int
	}{{"CPU threads", result.Reserved.CPUThreads, capacity.CPUThreads}, {"host RAM GiB", result.Reserved.HostRAMGiB, capacity.HostRAMGiB}, {"VRAM GiB", result.Reserved.VRAMGiB, capacity.VRAMGiB}} {
		if check.capacity > 0 && check.reserved > check.capacity {
			result.Conflicts = append(result.Conflicts, fmt.Sprintf("%s reserved %d exceeds capacity %d", check.name, check.reserved, check.capacity))
		}
	}
	result.Fits = len(result.Conflicts) == 0
	return result
}

// ValidCommit accepts a lower-case SHA-1 or SHA-256 Git identity.
func ValidCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	for _, char := range value {
		if char < '0' || char > '9' && char < 'a' || char > 'f' {
			return false
		}
	}
	return true
}

// NonCanonicalPathSeparator is the separator persisted worktree paths never carry; they use forward slashes on every host.
const NonCanonicalPathSeparator = "\\"
