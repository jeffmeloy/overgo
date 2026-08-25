package plan

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
)

const (
	workLeaseVersion uint16 = 1
	// WorkLeaseMediaType identifies work leases.
	WorkLeaseMediaType = "application/vnd.overgo.work-lease+json"
	// WorkLeaseSchema identifies the work-lease schema.
	WorkLeaseSchema = "overgo/work-lease/v1"
	// WorkLeaseAliasRoot scopes current worktree leases.
	WorkLeaseAliasRoot = "automation/worktree/"
)

// Resources is advisory capacity metadata; it never acquires hardware.
type Resources struct {
	CPUThreads   int  `json:"cpu_threads"`
	HostRAMGiB   int  `json:"host_ram_gib"`
	VRAMGiB      int  `json:"vram_gib"`
	GPUExclusive bool `json:"gpu_exclusive"`
}

// WorkLease: immutable lane assignment with CAS alias.
type WorkLease struct {
	Version       uint16    `json:"version"`
	Task          string    `json:"task"`
	Worktree      string    `json:"worktree"`
	Branch        string    `json:"branch"`
	Role          string    `json:"role"`
	TargetHead    string    `json:"target_head"`
	ConflictsWith []string  `json:"conflicts_with"`
	Resources     Resources `json:"resources"`
	ExpiresAt     string    `json:"expires_at"`
	// Optional experiment retry state.
	Experiment      artifact.ID `json:"experiment,omitzero"`
	Checkpoint      artifact.ID `json:"checkpoint,omitzero"`
	Retry           uint32      `json:"retry,omitempty"`
	PredictedWallNS uint64      `json:"predicted_wall_ns,omitempty"`
	ID              artifact.ID `json:"-"`
}

var workLeaseCodec = artifact.JSONDocumentCodec("work lease", artifact.KindEvidence, WorkLeaseMediaType, WorkLeaseSchema,
	canonicalizeWorkLease, func(value WorkLease) artifact.ID { return value.ID },
	func(value *WorkLease, id artifact.ID) { value.ID = id }, func(value WorkLease) WorkLease {
		value.ConflictsWith = slices.Clone(value.ConflictsWith)
		return value
	})

// RecordWorkLease: normalize and move the worktree alias by CAS.
func RecordWorkLease(ctx context.Context, repository artifact.Repository, data []byte) (WorkLease, error) {
	value, _, err := workLeaseCodec.Normalize(data)
	if err != nil {
		return WorkLease{}, err
	}
	current, exists, err := artifact.ResolveAlias(ctx, repository, workLeaseAlias(value.Worktree))
	if err != nil {
		return WorkLease{}, err
	}
	var previous *artifact.ID
	if exists {
		previous = &current
	}
	batch, err := workLeaseCodec.Batch("automation/work-lease/"+value.ID.String(), value, nil,
		[]artifact.AliasBinding{{Name: workLeaseAlias(value.Worktree), Target: value.ID, Previous: previous}})
	if err != nil {
		return WorkLease{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return value, err
}

// ReadWorkLease returns false for a non-lease artifact.
func ReadWorkLease(ctx context.Context, reader artifact.Reader, id artifact.ID) (WorkLease, bool, error) {
	return readTypedDocument(ctx, reader, id, workLeaseCodec.Contract, workLeaseCodec.Read)
}

// ParseWorkLease decodes one canonical work lease.
func ParseWorkLease(content []byte) (WorkLease, error) { return workLeaseCodec.Parse(content) }

func readTypedDocument[T any](ctx context.Context, reader artifact.Reader, id artifact.ID, contract artifact.DocumentContract, read func(context.Context, artifact.Reader, artifact.ID) (T, bool, error)) (T, bool, error) {
	var zero T
	descriptor, ok, err := reader.Artifact(ctx, id)
	if err != nil || !ok || descriptor.MediaType != contract.MediaType || descriptor.Schema != contract.Schema {
		return zero, false, err
	}
	return read(ctx, reader, id)
}

func workLeaseAlias(worktree string) string {
	digest := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(worktree))))
	return fmt.Sprintf("%s%x", WorkLeaseAliasRoot, digest)
}

func canonicalizeWorkLease(value *WorkLease) error {
	if value == nil || value.Version != workLeaseVersion || !validAutomationText(value.Task) ||
		!validAutomationText(value.Worktree) || strings.Contains(value.Worktree, "\\") ||
		!validAutomationText(value.Branch) || !validAutomationText(value.Role) || !validCommit(value.TargetHead) ||
		value.Resources.CPUThreads <= 0 || value.Resources.HostRAMGiB <= 0 || value.Resources.VRAMGiB < 0 ||
		value.Resources.GPUExclusive && value.Resources.VRAMGiB == 0 {
		return errors.New("plan: invalid work lease")
	}
	parsed, err := time.Parse(time.RFC3339Nano, value.ExpiresAt)
	if err != nil {
		return errors.New("plan: invalid work lease expiry")
	}
	value.ExpiresAt = parsed.UTC().Format(time.RFC3339Nano)
	if value.ConflictsWith == nil {
		return errors.New("plan: work lease conflicts must not be nil")
	}
	sort.Strings(value.ConflictsWith)
	value.ConflictsWith = slices.Compact(value.ConflictsWith)
	for _, item := range value.ConflictsWith {
		if !validAutomationText(item) {
			return errors.New("plan: invalid work lease conflict")
		}
	}
	if slices.Contains(value.ConflictsWith, value.Task) {
		return errors.New("plan: work lease cannot conflict with itself")
	}
	return nil
}

type ResourceAdvisory struct {
	ActiveTasks []string  `json:"active_tasks"`
	Reserved    Resources `json:"reserved"`
	Fits        bool      `json:"fits"`
	Conflicts   []string  `json:"conflicts"`
}

// AssessResources reports active reservations and collisions. Zero capacity
// means unknown, not zero available; this remains advice for the owner.
func AssessResources(now time.Time, capacity Resources, leases []WorkLease) ResourceAdvisory {
	active := make([]WorkLease, 0, len(leases))
	for _, lease := range leases {
		expires, err := time.Parse(time.RFC3339Nano, lease.ExpiresAt)
		if err == nil && expires.After(now) {
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
			case strings.EqualFold(left.Worktree, right.Worktree):
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

func validCommit(value string) bool {
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
