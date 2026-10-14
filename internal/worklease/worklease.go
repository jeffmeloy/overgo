// Package worklease owns work leases and dispatch claims: the immutable lane
// assignment records, their CAS aliases and the workspace claims that decide
// whether two leases collide. It sits below the plan so that operation
// runtimes consume leases without compiling the plan's readers.
package worklease

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
)

const (
	// Version is the advisory lease schema version.
	Version uint16 = 1
	// DispatchVersion is the dispatch-claim schema version; legacy retirement only sees v1.
	DispatchVersion uint16 = 2
	dispatchSchema         = "overgo/work-lease/v2"
	// MediaType identifies work leases.
	MediaType = "application/vnd.overgo.work-lease+json"
	// Schema identifies the work-lease schema.
	Schema = "overgo/work-lease/v1"
	// AliasRoot scopes current worktree leases.
	AliasRoot = "automation/worktree/"
)

// Resources is advisory capacity metadata; it never acquires hardware.
type Resources struct {
	CPUThreads   int  `json:"cpu_threads"`
	HostRAMGiB   int  `json:"host_ram_gib"`
	VRAMGiB      int  `json:"vram_gib"`
	GPUExclusive bool `json:"gpu_exclusive"`
}

// ValidResources accepts a positive CPU and RAM reservation with a non-negative VRAM one.
func ValidResources(resources Resources) bool {
	return resources.CPUThreads > 0 && resources.HostRAMGiB > 0 && resources.VRAMGiB >= 0 &&
		(!resources.GPUExclusive || resources.VRAMGiB > 0)
}

// WorkspaceClaims declares exact path access. WholeWorktree is the
// conservative representation for an absent or unresolvable scope.
type WorkspaceClaims struct {
	WholeWorktree bool     `json:"whole_worktree,omitzero"`
	Read          []string `json:"read,omitempty"`
	Write         []string `json:"write,omitempty"`
}

// Lease is one immutable lane assignment held through a CAS alias.
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

// Codec is the lease document codec: canonical form, identity and contracts.
var Codec = func() artifact.DocumentCodec[Lease] {
	codec := artifact.JSONDocumentCodec("work lease", artifact.KindEvidence, MediaType, Schema,
		canonicalize, func(value Lease) artifact.ID { return value.ID },
		func(value *Lease, id artifact.ID) { value.ID = id }, func(value Lease) Lease {
			value.ConflictsWith = slices.Clone(value.ConflictsWith)
			value.Claims.Read = slices.Clone(value.Claims.Read)
			value.Claims.Write = slices.Clone(value.Claims.Write)
			return value
		})
	codec.ContractFor = contractFor
	return codec
}()

func contractFor(value Lease) artifact.DocumentContract {
	schema := Schema
	if value.Version == DispatchVersion {
		schema = dispatchSchema
	}
	return artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: MediaType, Schema: schema}
}

func worktreeAlias(worktree string) string {
	return Alias(AliasRoot, strings.ToLower(strings.TrimSpace(worktree)))
}

func canonicalize(value *Lease) error {
	if value == nil || value.Version != Version && value.Version != DispatchVersion || !ValidAutomationText(value.Task) ||
		!ValidAutomationText(value.Worktree) || strings.Contains(value.Worktree, NonCanonicalPathSeparator) ||
		!ValidAutomationText(value.Branch) || !ValidAutomationText(value.Role) || !ValidCommit(value.TargetHead) {
		return errors.New("work lease: invalid work lease")
	}
	if !ValidResources(value.Resources) && !(value.Worker != "" && value.Resources == (Resources{})) {
		return errors.New("work lease: invalid work lease resources")
	}
	if value.Worker == "" {
		if value.Version != Version {
			return errors.New("work lease: dispatch schema requires a worker")
		}
		if value.Contract != "" || value.Acquisition != "" || value.Previous.Valid() {
			return errors.New("work lease: dispatch metadata requires a worker")
		}
		parsed, err := time.Parse(time.RFC3339Nano, value.ExpiresAt)
		if err != nil {
			return errors.New("work lease: invalid work lease expiry")
		}
		value.ExpiresAt = parsed.UTC().Format(time.RFC3339Nano)
	} else {
		if value.Version != DispatchVersion {
			return errors.New("work lease: dispatch ownership requires the v2 schema")
		}
		item, step, found := strings.Cut(value.Task, "/")
		if !ValidAutomationText(value.Worker) || value.Worker == UnassignedRole ||
			!found || !ValidPlanID(item) || !ValidPlanID(step) || value.ExpiresAt != "" ||
			value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
			return errors.New("work lease: invalid dispatch claim")
		}
		for _, digest := range []string{value.Contract, value.Acquisition} {
			decoded, err := hex.DecodeString(digest)
			if err != nil || len(decoded) != sha256.Size || digest != strings.ToLower(digest) {
				return errors.New("work lease: dispatch claim requires exact contract and acquisition digests")
			}
		}
	}
	if value.ConflictsWith == nil {
		return errors.New("work lease: work lease conflicts must not be nil")
	}
	slices.Sort(value.ConflictsWith)
	value.ConflictsWith = slices.Compact(value.ConflictsWith)
	for _, item := range value.ConflictsWith {
		if !ValidAutomationText(item) {
			return errors.New("work lease: invalid work lease conflict")
		}
	}
	if slices.Contains(value.ConflictsWith, value.Task) {
		return errors.New("work lease: work lease cannot conflict with itself")
	}
	canonicalizeClaims(&value.Claims)
	return nil
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

// Alias derives the alias of one identity under a root.
func Alias(prefix, identity string) string {
	digest := sha256.Sum256([]byte(identity))
	return fmt.Sprintf("%s%x", prefix, digest)
}

// NonCanonicalPathSeparator is the separator persisted worktree paths never carry; they use forward slashes on every host.
const NonCanonicalPathSeparator = "\\"
