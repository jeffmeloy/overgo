package plan

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/pathidentity"
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

type WorkspaceClaimMode string

const (
	WorkspaceClaimRead  WorkspaceClaimMode = "read"
	WorkspaceClaimWrite WorkspaceClaimMode = "write"
)

type WorkspaceClaim struct {
	Mode WorkspaceClaimMode `json:"mode"`
	Path string             `json:"path"`
}

// WorkspaceClaims declares exact path access. WholeWorktree is the
// conservative representation for an absent or unresolvable scope.
type WorkspaceClaims struct {
	WholeWorktree bool     `json:"whole_worktree,omitempty"`
	Read          []string `json:"read,omitempty"`
	Write         []string `json:"write,omitempty"`
}

// WorkLease: immutable lane assignment with CAS alias.
type WorkLease struct {
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
		value.Claims.Read = slices.Clone(value.Claims.Read)
		value.Claims.Write = slices.Clone(value.Claims.Write)
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
	canonicalizeClaims(&value.Claims)
	return nil
}

// ResolveWorkspaceClaims converts requested paths to live filesystem
// identities. Any uncertainty returns a whole-worktree claim.
func ResolveWorkspaceClaims(worktree string, read, write []string) (string, WorkspaceClaims) {
	root, err := pathidentity.Canonical(worktree)
	if err != nil {
		return filepath.ToSlash(filepath.Clean(worktree)), WorkspaceClaims{WholeWorktree: true}
	}
	claims := WorkspaceClaims{}
	resolve := func(values []string, destination *[]string) bool {
		for _, value := range values {
			candidate := value
			if !filepath.IsAbs(candidate) {
				candidate = filepath.Join(root, candidate)
			}
			contained, containErr := pathidentity.Contains(root, candidate)
			if containErr != nil || !contained {
				return false
			}
			absolute, absoluteErr := filepath.Abs(candidate)
			if absoluteErr != nil {
				return false
			}
			*destination = append(*destination, filepath.ToSlash(filepath.Clean(absolute)))
		}
		return true
	}
	if !resolve(read, &claims.Read) || !resolve(write, &claims.Write) || len(read)+len(write) == 0 {
		return filepath.ToSlash(root), WorkspaceClaims{WholeWorktree: true}
	}
	canonicalizeClaims(&claims)
	return filepath.ToSlash(root), claims
}

func canonicalizeClaims(claims *WorkspaceClaims) {
	if claims == nil {
		return
	}
	for _, values := range []*[]string{&claims.Read, &claims.Write} {
		for index, value := range *values {
			(*values)[index] = filepath.ToSlash(filepath.Clean(value))
		}
		sort.Slice(*values, func(i, j int) bool { return strings.ToLower((*values)[i]) < strings.ToLower((*values)[j]) })
		*values = slices.CompactFunc(*values, strings.EqualFold)
	}
	if len(claims.Read)+len(claims.Write) == 0 {
		claims.WholeWorktree = true
	}
	if claims.WholeWorktree {
		claims.Read, claims.Write = nil, nil
	}
}

// WorkspaceClaimOrder returns the single deterministic acquisition order.
func WorkspaceClaimOrder(claims WorkspaceClaims) []WorkspaceClaim {
	if claims.WholeWorktree {
		return []WorkspaceClaim{{Mode: WorkspaceClaimWrite, Path: "*"}}
	}
	result := make([]WorkspaceClaim, 0, len(claims.Read)+len(claims.Write))
	for _, path := range claims.Read {
		result = append(result, WorkspaceClaim{Mode: WorkspaceClaimRead, Path: path})
	}
	for _, path := range claims.Write {
		result = append(result, WorkspaceClaim{Mode: WorkspaceClaimWrite, Path: path})
	}
	sort.Slice(result, func(i, j int) bool {
		if !strings.EqualFold(result[i].Path, result[j].Path) {
			return strings.ToLower(result[i].Path) < strings.ToLower(result[j].Path)
		}
		return result[i].Mode < result[j].Mode
	})
	return result
}

// WorkspaceClaimsConflict reports write/read or write/write overlap.
func WorkspaceClaimsConflict(left, right WorkLease) bool {
	if !strings.EqualFold(filepath.Clean(left.Worktree), filepath.Clean(right.Worktree)) {
		return false
	}
	if left.Claims.WholeWorktree || right.Claims.WholeWorktree {
		return true
	}
	for _, write := range left.Claims.Write {
		for _, path := range append(slices.Clone(right.Claims.Read), right.Claims.Write...) {
			if claimPathsOverlap(write, path) {
				return true
			}
		}
	}
	for _, write := range right.Claims.Write {
		for _, read := range left.Claims.Read {
			if claimPathsOverlap(write, read) {
				return true
			}
		}
	}
	return false
}

func claimPathsOverlap(left, right string) bool {
	left, right = strings.TrimSuffix(filepath.ToSlash(filepath.Clean(left)), "/"), strings.TrimSuffix(filepath.ToSlash(filepath.Clean(right)), "/")
	return strings.EqualFold(left, right) || strings.HasPrefix(strings.ToLower(left), strings.ToLower(right)+"/") || strings.HasPrefix(strings.ToLower(right), strings.ToLower(left)+"/")
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
