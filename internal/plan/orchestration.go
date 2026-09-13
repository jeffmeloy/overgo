package plan

import (
	"context"
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
	"overgo/internal/pathidentity"
)

const (
	workLeaseVersion uint16 = 1
	// Dispatch ownership has a separate wire schema; legacy retirement only sees v1.
	dispatchLeaseVersion uint16 = 2
	dispatchLeaseSchema         = "overgo/work-lease/v2"
	// WorkLeaseMediaType identifies work leases.
	WorkLeaseMediaType = "application/vnd.overgo.work-lease+json"
	// WorkLeaseSchema identifies the work-lease schema.
	WorkLeaseSchema = "overgo/work-lease/v1"
	// WorkLeaseAliasRoot scopes current worktree leases.
	WorkLeaseAliasRoot  = "automation/worktree/"
	workTaskAliasRoot   = "automation/dispatch/task/"
	workWorkerAliasRoot = "automation/dispatch/worker/"
)

// Resources is advisory capacity metadata; it never acquires hardware.
type Resources struct {
	CPUThreads   int  `json:"cpu_threads"`
	HostRAMGiB   int  `json:"host_ram_gib"`
	VRAMGiB      int  `json:"vram_gib"`
	GPUExclusive bool `json:"gpu_exclusive"`
}

func validLeaseResources(resources Resources) bool {
	return resources.CPUThreads > 0 && resources.HostRAMGiB > 0 && resources.VRAMGiB >= 0 &&
		(!resources.GPUExclusive || resources.VRAMGiB > 0)
}

// WorkspaceClaimMode distinguishes read from write path access.
type WorkspaceClaimMode string

const (
	// WorkspaceClaimRead claims shared read access to a path.
	WorkspaceClaimRead WorkspaceClaimMode = "read"
	// WorkspaceClaimWrite claims exclusive write access to a path.
	WorkspaceClaimWrite WorkspaceClaimMode = "write"
)

// WorkspaceClaim is one mode-qualified path in the deterministic
// acquisition order.
type WorkspaceClaim struct {
	Mode WorkspaceClaimMode `json:"mode"`
	Path string             `json:"path"`
}

// WorkspaceClaims declares exact path access. WholeWorktree is the
// conservative representation for an absent or unresolvable scope.
type WorkspaceClaims struct {
	WholeWorktree bool     `json:"whole_worktree,omitzero"`
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

// NewWorkLease validates and identifies one immutable lease declaration.
func NewWorkLease(value WorkLease) (WorkLease, error) {
	if value.Worker != "" {
		value.Version = dispatchLeaseVersion
		return workLeaseCodec.New(value)
	}
	return workLeaseCodec.NewInitial(value)
}

var workLeaseCodec = func() artifact.DocumentCodec[WorkLease] {
	codec := artifact.JSONDocumentCodec("work lease", artifact.KindEvidence, WorkLeaseMediaType, WorkLeaseSchema,
		canonicalizeWorkLease, func(value WorkLease) artifact.ID { return value.ID },
		func(value *WorkLease, id artifact.ID) { value.ID = id }, func(value WorkLease) WorkLease {
			value.ConflictsWith = slices.Clone(value.ConflictsWith)
			value.Claims.Read = slices.Clone(value.Claims.Read)
			value.Claims.Write = slices.Clone(value.Claims.Write)
			return value
		})
	codec.ContractFor = workLeaseContract
	return codec
}()

func workLeaseContract(value WorkLease) artifact.DocumentContract {
	schema := WorkLeaseSchema
	if value.Version == dispatchLeaseVersion {
		schema = dispatchLeaseSchema
	}
	return artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: WorkLeaseMediaType, Schema: schema}
}

// WorkLeaseContracts is the shared query contract for advisory leases and dispatch claims.
func WorkLeaseContracts() []artifact.DocumentContract {
	return []artifact.DocumentContract{workLeaseContract(WorkLease{Version: workLeaseVersion}), workLeaseContract(WorkLease{Version: dispatchLeaseVersion})}
}

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
	if exists && current == value.ID {
		return value, ResolveWorkLeaseOwner(ctx, repository, value)
	}
	if value.Worker != "" {
		return recordDispatchLease(ctx, repository, value)
	}
	if exists {
		owner, found, err := ReadWorkLease(ctx, repository, current)
		if err != nil {
			return WorkLease{}, err
		}
		if found && owner.Worker != "" {
			return WorkLease{}, errors.New("plan: advisory lease cannot replace a dispatch claim")
		}
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
	descriptor, found, err := reader.Artifact(ctx, id)
	if err != nil || !found || descriptor.MediaType != WorkLeaseMediaType || descriptor.Schema != WorkLeaseSchema && descriptor.Schema != dispatchLeaseSchema {
		return WorkLease{}, false, err
	}
	return workLeaseCodec.Read(ctx, reader, id)
}

// ParseWorkLease decodes one canonical work lease.
func ParseWorkLease(content []byte) (WorkLease, error) { return workLeaseCodec.Parse(content) }

// ValidateIdentity checks the lease's canonical form and content-addressed identity.
func (lease WorkLease) ValidateIdentity() error { return workLeaseCodec.ValidateIdentity(lease) }

// WorkLeaseAlias names the existing CAS ownership binding for a worktree.
func WorkLeaseAlias(worktree string) string { return workLeaseAlias(worktree) }

// ResolveWorkLeaseOwner requires the exact lease to remain the current CAS owner.
func ResolveWorkLeaseOwner(ctx context.Context, reader artifact.Reader, lease WorkLease) error {
	if err := lease.ValidateIdentity(); err != nil {
		return err
	}
	for _, alias := range workLeaseAliases(lease) {
		current, found, err := reader.ResolveAlias(ctx, alias)
		if err != nil {
			return err
		}
		if !found || current != lease.ID {
			return fmt.Errorf("plan: work lease is not the current owner of %s", alias)
		}
	}
	return nil
}

func readTypedDocument[T any](ctx context.Context, reader artifact.Reader, id artifact.ID, contract artifact.DocumentContract, read func(context.Context, artifact.Reader, artifact.ID) (T, bool, error)) (T, bool, error) {
	var zero T
	descriptor, ok, err := reader.Artifact(ctx, id)
	if err != nil || !ok || descriptor.MediaType != contract.MediaType || descriptor.Schema != contract.Schema {
		return zero, false, err
	}
	return read(ctx, reader, id)
}

func workLeaseAlias(worktree string) string {
	return leaseAlias(WorkLeaseAliasRoot, strings.ToLower(strings.TrimSpace(worktree)))
}

func canonicalizeWorkLease(value *WorkLease) error {
	if value == nil || value.Version != workLeaseVersion && value.Version != dispatchLeaseVersion || !validAutomationText(value.Task) ||
		!validAutomationText(value.Worktree) || strings.Contains(value.Worktree, "\\") ||
		!validAutomationText(value.Branch) || !validAutomationText(value.Role) || !validCommit(value.TargetHead) {
		return errors.New("plan: invalid work lease")
	}
	if !validLeaseResources(value.Resources) && !(value.Worker != "" && value.Resources == (Resources{})) {
		return errors.New("plan: invalid work lease resources")
	}
	if value.Worker == "" {
		if value.Version != workLeaseVersion {
			return errors.New("plan: dispatch schema requires a worker")
		}
		if value.Contract != "" || value.Acquisition != "" || value.Previous.Valid() {
			return errors.New("plan: dispatch metadata requires a worker")
		}
		parsed, err := time.Parse(time.RFC3339Nano, value.ExpiresAt)
		if err != nil {
			return errors.New("plan: invalid work lease expiry")
		}
		value.ExpiresAt = parsed.UTC().Format(time.RFC3339Nano)
	} else {
		if value.Version != dispatchLeaseVersion {
			return errors.New("plan: dispatch ownership requires the v2 schema")
		}
		item, step, found := strings.Cut(value.Task, "/")
		if !validAutomationText(value.Worker) || value.Worker == UnassignedRole ||
			!found || !validPlanID(item) || !validPlanID(step) || value.ExpiresAt != "" ||
			value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
			return errors.New("plan: invalid dispatch claim")
		}
		for _, digest := range []string{value.Contract, value.Acquisition} {
			decoded, err := hex.DecodeString(digest)
			if err != nil || len(decoded) != sha256.Size || digest != strings.ToLower(digest) {
				return errors.New("plan: dispatch claim requires exact contract and acquisition digests")
			}
		}
	}
	if value.ConflictsWith == nil {
		return errors.New("plan: work lease conflicts must not be nil")
	}
	slices.Sort(value.ConflictsWith)
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

func workLeaseAliases(lease WorkLease) []string {
	aliases := []string{workLeaseAlias(lease.Worktree)}
	if lease.Worker != "" {
		aliases = append(aliases, leaseAlias(workTaskAliasRoot, lease.Task), leaseAlias(workWorkerAliasRoot, lease.Worker))
	}
	return aliases
}

func leaseAlias(prefix, identity string) string {
	digest := sha256.Sum256([]byte(identity))
	return fmt.Sprintf("%s%x", prefix, digest)
}

// All aliases move in one transaction; distinct worktrees cannot claim the same task.
func recordDispatchLease(ctx context.Context, repository artifact.Repository, value WorkLease) (WorkLease, error) {
	var previous *artifact.ID
	var lineage []artifact.Lineage
	if value.Previous.Valid() {
		owner, found, err := ReadWorkLease(ctx, repository, value.Previous)
		if err != nil {
			return WorkLease{}, err
		}
		if !found || owner.Worker != value.Worker || owner.Task != value.Task ||
			owner.Role != value.Role || !strings.EqualFold(owner.Worktree, value.Worktree) {
			return WorkLease{}, errors.New("plan: claim renewal requires the same worker, task, role and worktree")
		}
		previous = &value.Previous
		lineage = []artifact.Lineage{{Child: value.ID, Parent: owner.ID, Relation: artifact.RelationDerivedFrom}}
	}
	var aliases []artifact.AliasBinding
	for _, name := range workLeaseAliases(value) {
		aliases = append(aliases, artifact.AliasBinding{Name: name, Target: value.ID, Previous: previous})
	}
	batch, err := workLeaseCodec.Batch("automation/work-lease/"+value.ID.String(), value, lineage, aliases)
	if err != nil {
		return WorkLease{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return WorkLease{}, err
	}
	return value, ResolveWorkLeaseOwner(ctx, repository, value)
}

// ReleaseBatch retires exact ownership with CAS, for atomic composition with
// completion evidence. Lease contents, results and outstanding checks survive.
func (lease WorkLease) ReleaseBatch(worker, reason string) (artifact.Batch, error) {
	if reason != "completed" && reason != "cancelled" && reason != "handoff" {
		return artifact.Batch{}, errors.New("plan: claim release requires completed, cancelled or handoff")
	}
	if err := lease.ValidateIdentity(); err != nil {
		return artifact.Batch{}, err
	}
	if worker == "" || lease.Worker != worker {
		return artifact.Batch{}, errors.New("plan: claim release requires its exact worker and lease")
	}
	id := lease.ID
	batch := artifact.Batch{Key: "automation/work-lease/release/" + id.String() + "/" + reason}
	for _, name := range workLeaseAliases(lease) {
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: name, Target: id, Previous: &id, Remove: true})
	}
	return batch, nil
}
