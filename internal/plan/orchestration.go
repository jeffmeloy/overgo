package plan

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	WorkLeaseVersion   uint16 = 1
	WorkLeaseMediaType        = "application/vnd.overgo.work-lease+json"
	WorkLeaseSchema           = "overgo/work-lease/v1"
	workLeaseAliasRoot        = "automation/worktree/"
)

// ResourceRequest is advisory capacity metadata; it never acquires hardware.
type ResourceRequest struct {
	CPUThreads   int  `json:"cpu_threads"`
	HostRAMGiB   int  `json:"host_ram_gib"`
	VRAMGiB      int  `json:"vram_gib"`
	GPUExclusive bool `json:"gpu_exclusive"`
}

// WorkLease records an owner-approved lane assignment. The worktree alias is
// compare-and-set in RepoDB, but the document itself is immutable evidence.
type WorkLease struct {
	Version       uint16          `json:"version"`
	Task          string          `json:"task"`
	Worktree      string          `json:"worktree"`
	Branch        string          `json:"branch"`
	Role          string          `json:"role"`
	TargetHead    string          `json:"target_head"`
	DependsOn     []string        `json:"depends_on"`
	ConflictsWith []string        `json:"conflicts_with"`
	Resources     ResourceRequest `json:"resources"`
	EvidenceLanes []string        `json:"evidence_lanes"`
	ExpiresAt     string          `json:"expires_at"`
	ID            artifact.ID     `json:"-"`
}

var workLeaseCodec = artifact.JSONDocumentCodec("work lease", artifact.KindEvidence, WorkLeaseMediaType, WorkLeaseSchema,
	canonicalizeWorkLease, func(value WorkLease) artifact.ID { return value.ID },
	func(value *WorkLease, id artifact.ID) { value.ID = id }, func(value WorkLease) WorkLease {
		value.DependsOn = slices.Clone(value.DependsOn)
		value.ConflictsWith = slices.Clone(value.ConflictsWith)
		value.EvidenceLanes = slices.Clone(value.EvidenceLanes)
		return value
	})

func NewWorkLease(value WorkLease) (WorkLease, error) {
	value.Version = WorkLeaseVersion
	return workLeaseCodec.New(value)
}

func NormalizeWorkLease(data []byte) (WorkLease, error) {
	value, _, err := workLeaseCodec.Normalize(data)
	return value, err
}

func ParseWorkLease(data []byte) (WorkLease, error)        { return workLeaseCodec.Parse(data) }
func (value WorkLease) Content() (artifact.Content, error) { return workLeaseCodec.Content(value) }
func (value WorkLease) ValidateIdentity() error            { return workLeaseCodec.ValidateIdentity(value) }

// WorkLeaseBatch atomically records a lease and moves its worktree alias. A
// nil previous ID acquires an unbound worktree; a non-nil ID is RepoDB CAS.
func WorkLeaseBatch(value WorkLease, previous *artifact.ID) (artifact.Batch, error) {
	content, err := value.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch("automation/work-lease/"+value.ID.String(), []artifact.Content{content}, nil,
		[]artifact.AliasBinding{{Name: WorkLeaseAlias(value.Worktree), Target: value.ID, Previous: previous}})
}

func WorkLeaseAlias(worktree string) string {
	digest := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(worktree))))
	return fmt.Sprintf("%s%x", workLeaseAliasRoot, digest)
}

func canonicalizeWorkLease(value *WorkLease) error {
	if value == nil || value.Version != WorkLeaseVersion || !textcheck.Bounded(value.Task, 2048, "\x00\r\n") ||
		!textcheck.Bounded(value.Worktree, 2048, "\x00\r\n") || strings.Contains(value.Worktree, "\\") ||
		!textcheck.Bounded(value.Branch, 2048, "\x00\r\n") || !textcheck.Bounded(value.Role, 2048, "\x00\r\n") || !validCommit(value.TargetHead) ||
		value.Resources.CPUThreads <= 0 || value.Resources.HostRAMGiB <= 0 || value.Resources.VRAMGiB < 0 ||
		value.Resources.GPUExclusive && value.Resources.VRAMGiB == 0 {
		return errors.New("plan: invalid work lease")
	}
	parsed, err := time.Parse(time.RFC3339Nano, value.ExpiresAt)
	if err != nil {
		return errors.New("plan: invalid work lease expiry")
	}
	value.ExpiresAt = parsed.UTC().Format(time.RFC3339Nano)
	for _, values := range []*[]string{&value.DependsOn, &value.ConflictsWith, &value.EvidenceLanes} {
		if *values == nil {
			return errors.New("plan: work lease lists must not be nil")
		}
		sort.Strings(*values)
		*values = slices.Compact(*values)
		for _, item := range *values {
			if !textcheck.Bounded(item, 2048, "\x00\r\n") {
				return errors.New("plan: invalid work lease list value")
			}
		}
	}
	if slices.Contains(value.DependsOn, value.Task) || slices.Contains(value.ConflictsWith, value.Task) {
		return errors.New("plan: work lease cannot depend on or conflict with itself")
	}
	return nil
}

type ResourceCapacity struct {
	CPUThreads int `json:"cpu_threads,omitempty"`
	HostRAMGiB int `json:"host_ram_gib,omitempty"`
	VRAMGiB    int `json:"vram_gib,omitempty"`
}

type ResourceAdvisory struct {
	ActiveTasks []string        `json:"active_tasks"`
	Reserved    ResourceRequest `json:"reserved"`
	Fits        bool            `json:"fits"`
	Conflicts   []string        `json:"conflicts"`
}

// AssessResources reports active reservations and collisions. Zero capacity
// means unknown, not zero available; this remains advice for the owner.
func AssessResources(now time.Time, capacity ResourceCapacity, leases []WorkLease) ResourceAdvisory {
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

type MergeEligibilityInput struct {
	CurrentTargetHead string
	CandidateHead     string
	Lease             WorkLease
	ActiveLease       artifact.ID
	WorktreeClean     bool
	Conflicts         []string
	ReviewVerdict     artifact.ID
	RequiredEvidence  []artifact.ID
	ObservedEvidence  []artifact.ID
}

type MergeEligibility struct {
	TargetHead            string   `json:"target_head"`
	CandidateHead         string   `json:"candidate_head"`
	Eligible              bool     `json:"eligible"`
	OwnerDecisionRequired bool     `json:"owner_decision_required"`
	Reasons               []string `json:"reasons"`
}

// AssessMergeEligibility produces an advisory packet only. The owner still
// chooses whether and when to merge or promote the candidate.
func AssessMergeEligibility(input MergeEligibilityInput) MergeEligibility {
	result := MergeEligibility{TargetHead: input.CurrentTargetHead, CandidateHead: input.CandidateHead, OwnerDecisionRequired: true, Reasons: []string{}}
	if !validCommit(input.CurrentTargetHead) || !validCommit(input.CandidateHead) || input.CurrentTargetHead == input.CandidateHead {
		result.Reasons = append(result.Reasons, "target and candidate heads must be distinct valid commits")
	}
	if input.Lease.TargetHead != input.CurrentTargetHead {
		result.Reasons = append(result.Reasons, "target head moved after the lease was recorded")
	}
	if input.Lease.ValidateIdentity() != nil {
		result.Reasons = append(result.Reasons, "worktree lease identity is invalid")
	}
	if input.ActiveLease != input.Lease.ID {
		result.Reasons = append(result.Reasons, "worktree lease is not current")
	}
	if !input.WorktreeClean {
		result.Reasons = append(result.Reasons, "worktree is dirty")
	}
	if len(input.Conflicts) > 0 {
		result.Reasons = append(result.Reasons, "worktree or resource conflicts remain")
	}
	if input.ReviewVerdict.Kind() != artifact.KindEvidence {
		result.Reasons = append(result.Reasons, "admitted SQA verdict is absent")
	}
	observed := make(map[artifact.ID]bool, len(input.ObservedEvidence))
	for _, id := range input.ObservedEvidence {
		observed[id] = true
	}
	for _, id := range input.RequiredEvidence {
		if id.Kind() != artifact.KindEvidence || !observed[id] {
			result.Reasons = append(result.Reasons, "required evidence is absent: "+id.String())
		}
	}
	result.Eligible = len(result.Reasons) == 0
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
