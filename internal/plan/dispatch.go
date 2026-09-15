package plan

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/worklease"
)

// Dispatch is the current row as data: the harness reads its fields and
// Line is the one-line prose rendered from the same fields.
type Dispatch struct {
	Stop      *StopStatus      `json:"stop,omitempty"`
	Complete  bool             `json:"complete"`
	Item      string           `json:"item,omitzero"`
	Step      string           `json:"step,omitzero"`
	ItemTitle string           `json:"item_title,omitzero"`
	StepTitle string           `json:"step_title,omitzero"`
	Verify    string           `json:"verify,omitzero"`
	Line      string           `json:"line"`
	Claim     *worklease.Lease `json:"claim,omitempty"`
	ClaimID   artifact.ID      `json:"claim_id,omitzero"`
	Waiting   string           `json:"waiting,omitzero"`
}

// DispatchCachePath keeps the last resolved dispatch beside the inputs it
// was resolved from, so a turn's several readers resolve the completion
// authority once.
const DispatchCachePath = "docs/.dispatch"

// Shared Git revision query used by dispatch and completion authority.
const gitRevisionCommand = "rev-parse"

// Git names the current worktree revision through HEAD.
const gitHeadRevision = "HEAD"

// DispatchRequest separates read-only inspection from executable acquisition.
// Worker is a stable session identity, never the shared role or command PID.
type DispatchRequest struct {
	Mode        string
	Maintenance string
	Role        string
	Worker      string
	Acquire     bool
	Reference   string
}

// AutomationWorkerEnvironment is inherited by a driver and its gate subprocesses.
const AutomationWorkerEnvironment = "OVERGO_AUTOMATION_WORKER"

func dispatchWorker(explicit string) (string, error) {
	worker := cmp.Or(strings.TrimSpace(explicit), strings.TrimSpace(os.Getenv(AutomationWorkerEnvironment)))
	if worker != "" && (!worklease.ValidAutomationText(worker) || worker == worklease.UnassignedRole) {
		return "", errors.New("plan: worker must be a stable distinct session identity")
	}
	return worker, nil
}

// DispatchOf resolves the current row for role under a resolved authority.
func DispatchOf(document Plan, role string, authority CompletionAuthority) Dispatch {
	item, step, ok := Current(document, role, authority)
	if !ok {
		return finishDispatch(Dispatch{Complete: true})
	}
	return stepDispatch(item, step, nil)
}

func stepDispatch(item Item, step Step, claim *worklease.Lease) Dispatch {
	return finishDispatch(Dispatch{Item: item.ID, Step: step.ID, ItemTitle: item.Title, StepTitle: step.Title, Verify: step.Verify, Claim: claim})
}

func finishDispatch(dispatch Dispatch) Dispatch {
	if dispatch.Claim != nil {
		dispatch.ClaimID = dispatch.Claim.ID
	}
	dispatch.Line = dispatchLine(dispatch)
	return dispatch
}

func dispatchLine(dispatch Dispatch) string {
	switch {
	case dispatch.Waiting != "":
		return "plan waiting: " + dispatch.Waiting
	case dispatch.Complete:
		return "plan complete: every item is done"
	case dispatch.Step == ".":
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", dispatch.Item, dispatch.ItemTitle)
	default:
		return fmt.Sprintf("%s / %s: %s -- %s", dispatch.Item, dispatch.Step, dispatch.ItemTitle, dispatch.StepTitle)
	}
}

// dispatchCache binds a dispatch to every input it was resolved from.
type dispatchCache struct {
	Head          string            `json:"head"`
	PlanDigest    string            `json:"plan_digest"`
	StoreHead     artifact.CommitID `json:"store_head"`
	StoreSequence uint64            `json:"store_sequence"`
	Role          string            `json:"role"`
	Worker        string            `json:"worker"`
	Reference     string            `json:"reference"`
	Policy        string            `json:"policy"`
	Dispatch      Dispatch          `json:"dispatch"`
}

// ResolveDispatch resolves the current row for the repository at root and
// role, reusing the cached dispatch when HEAD, the plan bytes, the store
// head and the role are the ones it was resolved from.
func ResolveDispatch(ctx context.Context, root string, request DispatchRequest) (dispatch Dispatch, err error) {
	role, err := AutomationRole(request.Role)
	if err != nil {
		return Dispatch{}, err
	}
	worker, err := dispatchWorker(request.Worker)
	if err != nil {
		return Dispatch{}, err
	}
	if request.Acquire && worker == "" {
		return Dispatch{}, fmt.Errorf("plan: executable dispatch requires -worker <stable-session-id> or %s", AutomationWorkerEnvironment)
	}
	repository, head, err := resolveCompletionRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return Dispatch{}, err
	}
	if request.Acquire {
		if err := processcontrol.RequireCampaign(repository, worker); err != nil {
			return Dispatch{}, err
		}
		lock, lockErr := authoritylock.Acquire(repository)
		if errors.Is(lockErr, processlock.ErrBusy) {
			return finishDispatch(Dispatch{Waiting: "another plan or gate mutation owns this worktree; resume after its release"}), nil
		}
		if lockErr != nil {
			return Dispatch{}, lockErr
		}
		defer func() { err = errors.Join(err, lock.Close()) }()
		// HEAD and document are read under the mutation owner.
		_, head, err = resolveCompletionRevision(ctx, repository, gitHeadRevision)
		if err != nil {
			return Dispatch{}, err
		}
	}
	openStore := overgodb.OpenReadOnly
	if request.Acquire {
		openStore = overgodb.Open
	}
	store, err := openStore(filepath.Join(repository, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return Dispatch{}, err
	}
	defer func() { err = errors.Join(err, store.Close()) }()
	if request.Acquire {
		if _, err := gitauthority.RequireRegisteredWorktreeStore(ctx, repository, filepath.Join(repository, gitauthority.CanonicalOvergoDBDirectory), head); err != nil {
			return Dispatch{}, err
		}
	}
	mode, err := ExecutionMode(request.Mode)
	if err != nil {
		return Dispatch{}, err
	}
	stop, err := ReadStop(ctx, store, repository, head, mode)
	if err != nil {
		return Dispatch{}, err
	}
	grant := cmp.Or(request.Maintenance, strings.TrimSpace(os.Getenv(AutomationMaintenanceEnvironment)))
	if stop.Blocked && grant == "" {
		return finishDispatch(Dispatch{Stop: &stop, Waiting: stop.String()}), nil
	}
	planPath := filepath.Join(repository, filepath.FromSlash(Path))
	raw, err := os.ReadFile(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	// The cache takes the plan file's own permissions.
	planInfo, err := os.Stat(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	digest := sha256.Sum256(raw)
	projectStop := func(value Dispatch) Dispatch {
		if stop.State != stopAbsent {
			value.Stop = &stop
		}
		if stopErr := stop.RequireExecution(ctx, store, role, worker, mode, value.Item+"/"+value.Step, grant); stopErr != nil {
			value.Waiting = stopErr.Error()
			value = finishDispatch(value)
		}
		return value
	}
	storeHead, sequence := store.Head()
	key := dispatchCache{Head: head, PlanDigest: hex.EncodeToString(digest[:]), StoreHead: storeHead, StoreSequence: sequence, Role: role, Worker: worker, Reference: request.Reference, Policy: "claimed-prerequisite-frontier/v1"}
	cachePath := filepath.Join(repository, filepath.FromSlash(DispatchCachePath))
	if cached, ok := cachedDispatch(cachePath, key); ok && !request.Acquire {
		return projectStop(cached), nil
	}
	document, err := Load(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	authority, err := ResolveCompletionAuthority(ctx, repository, gitHeadRevision, document, store)
	if err != nil {
		return Dispatch{}, err
	}
	key.Dispatch, err = selectDispatch(ctx, store, document, role, filepath.ToSlash(repository), worker, request.Reference, authority)
	if err != nil {
		return Dispatch{}, err
	}
	if stopped := projectStop(key.Dispatch); stopped.Waiting != "" && stopped.Stop != nil {
		return stopped, nil
	}
	attempted := map[string]bool{}
	for request.Acquire && key.Dispatch.Waiting == "" && !key.Dispatch.Complete && key.Dispatch.Claim == nil {
		if key.Dispatch.Step == "." {
			return Dispatch{}, errors.New("plan: define the empty rung before claiming executable work")
		}
		item, step, _ := dispatchStep(document, key.Dispatch.Item+"/"+key.Dispatch.Step)
		reference := item.ID + "/" + step.ID
		if attempted[reference] {
			return finishDispatch(Dispatch{Waiting: "ownership changed repeatedly; resume after claim release"}), nil
		}
		attempted[reference] = true
		contract, contractErr := dispatchContract(item, step)
		if contractErr != nil {
			return Dispatch{}, contractErr
		}
		branch, branchErr := gitCompletionCommand(ctx, repository, gitRevisionCommand, "--abbrev-ref", gitHeadRevision)
		if branchErr != nil {
			return Dispatch{}, branchErr
		}
		acquisition, _ := store.Head()
		lease, leaseErr := worklease.New(worklease.Lease{Task: reference, Worktree: filepath.ToSlash(repository), Branch: strings.TrimSpace(string(branch)), Role: role, Worker: worker, TargetHead: head, Contract: contract, Acquisition: acquisition.String(), ConflictsWith: []string{}, Claims: worklease.WorkspaceClaims{WholeWorktree: true}})
		if leaseErr != nil {
			return Dispatch{}, leaseErr
		}
		content, contentErr := worklease.Codec.Content(lease)
		if contentErr != nil {
			return Dispatch{}, contentErr
		}
		claimed, claimErr := worklease.Record(ctx, store, content.Data)
		if claimErr != nil {
			if !errors.Is(claimErr, overgodb.ErrAliasConflict) {
				return Dispatch{}, claimErr
			}
			// Retry only a newly eligible row; never spin on the same claim.
			key.Dispatch, err = selectDispatch(ctx, store, document, role, filepath.ToSlash(repository), worker, request.Reference, authority)
			if err != nil {
				return Dispatch{}, err
			}
			continue
		}
		key.Dispatch.Claim = &claimed
		key.Dispatch.ClaimID = claimed.ID
		key.StoreHead, key.StoreSequence = store.Head()
	}
	// The cache is a convenience: a write that fails leaves the next reader resolving again.
	_ = jsonfile.Write(cachePath, key, planInfo.Mode().Perm())
	return projectStop(key.Dispatch), nil
}

// cachedDispatch returns the cached dispatch when every input matches key.
func cachedDispatch(path string, key dispatchCache) (Dispatch, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Dispatch{}, false
	}
	var cached dispatchCache
	if err := json.Unmarshal(raw, &cached); err != nil {
		return Dispatch{}, false
	}
	if cached.Dispatch.Claim != nil {
		claim, err := worklease.New(*cached.Dispatch.Claim)
		if err != nil || claim.ID != cached.Dispatch.ClaimID {
			return Dispatch{}, false
		}
		cached.Dispatch.Claim = &claim
	} else if cached.Dispatch.ClaimID.Valid() {
		return Dispatch{}, false
	}
	key.Dispatch = cached.Dispatch
	if cached != key || cached.Dispatch.Line == "" {
		return Dispatch{}, false
	}
	return cached.Dispatch, true
}

// Bind only this row and its item metadata. Finishing an unrelated sibling
// must not invalidate an active worker's contract.
func dispatchContract(item Item, step Step) (string, error) {
	item.Steps = []Step{step}
	digest, err := completionContractDigest(item)
	return hex.EncodeToString(digest[:]), err
}

func dispatchStep(document Plan, ref string) (Item, Step, bool) {
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if item.ID+"/"+step.ID == ref {
				return item, step, true
			}
		}
	}
	return Item{}, Step{}, false
}

// Visit prerequisites before the blocked objective's later siblings; retain
// document order otherwise. This changes scheduling, never dependency evidence.
func dispatchPriority(document Plan, role string, authority CompletionAuthority) ([]Ref, error) {
	frontier, err := ReadyFrontier(document, authority)
	if err != nil {
		return nil, err
	}
	ready := make(map[string]bool, len(frontier))
	for _, ref := range frontier {
		ready[ref.String()] = true
	}
	role = normalizedRole(role)
	if role == worklease.UnassignedRole && document.Lane != "" {
		role = document.Lane
	}
	eligible := func(item Item) bool {
		return item.Owner == "" || role != worklease.UnassignedRole && item.Owner == role
	}
	visited := map[string]bool{}
	var ordered []Ref
	var visit func(Item, Step)
	visit = func(item Item, step Step) {
		ref := Ref{Item: item.ID, Step: step.ID}
		if visited[ref.String()] || item.Status != StatusOpen || step.Status != StatusOpen {
			return
		}
		visited[ref.String()] = true
		if ready[ref.String()] {
			if eligible(item) {
				ordered = append(ordered, ref)
			}
			return
		}
		for _, dependency := range step.DependsOn {
			if parent, prerequisite, found := dispatchStep(document, dependency); found {
				visit(parent, prerequisite)
			}
		}
	}
	owners := []string{""}
	if role != worklease.UnassignedRole {
		owners = []string{role, ""}
	}
	for _, owner := range owners {
		for _, item := range document.Items {
			if item.Owner == owner {
				for _, step := range item.Steps {
					visit(item, step)
				}
			}
		}
	}
	return ordered, nil
}

func readDispatchAlias(ctx context.Context, reader artifact.Reader, alias string) (*worklease.Lease, error) {
	id, found, err := reader.ResolveAlias(ctx, alias)
	if err != nil || !found {
		return nil, err
	}
	lease, found, err := worklease.Read(ctx, reader, id)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("plan: ownership alias %s has no readable lease; explicit recovery required", alias)
	}
	return &lease, nil
}

// ValidateClaimedPlan refuses contract edits beneath active work. Release or
// explicit handoff precedes amendment; unrelated steps remain editable.
func ValidateClaimedPlan(ctx context.Context, reader artifact.Reader, before, after Plan) error {
	if err := Validate(after); err != nil {
		return err
	}
	visited := map[string]bool{}
	for _, document := range []Plan{before, after} {
		for _, item := range document.Items {
			for _, step := range item.Steps {
				ref := item.ID + "/" + step.ID
				if visited[ref] {
					continue
				}
				visited[ref] = true
				oldItem, oldStep, oldFound := dispatchStep(before, ref)
				newItem, newStep, newFound := dispatchStep(after, ref)
				oldContract, err := dispatchContract(oldItem, oldStep)
				if err != nil {
					return err
				}
				newContract, err := dispatchContract(newItem, newStep)
				if err != nil {
					return err
				}
				if oldFound == newFound && oldContract == newContract {
					continue
				}
				lease, err := readDispatchAlias(ctx, reader, worklease.Alias(worklease.TaskAliasRoot, ref))
				if err != nil {
					return err
				}
				if lease != nil {
					return fmt.Errorf("plan: %s is claimed by worker %s; release exact claim %s for handoff before editing its contract", ref, lease.Worker, lease.ID)
				}
			}
		}
	}
	return nil
}

func validateDispatchContract(document Plan, lease worklease.Lease, role string, authority CompletionAuthority) error {
	if normalizedRole(role) != lease.Role {
		return errors.New("plan: claim belongs to a different dispatch role")
	}
	item, step, found := dispatchStep(document, lease.Task)
	if !found || item.Status != StatusOpen || step.Status != StatusOpen {
		return errors.New("plan: claimed row is no longer open; reconcile its completion or handoff")
	}
	contract, err := dispatchContract(item, step)
	if err != nil {
		return err
	}
	if contract != lease.Contract {
		return errors.New("plan: claimed row contract changed; explicit handoff required")
	}
	ready, err := dispatchPriority(document, role, authority)
	if err != nil {
		return err
	}
	for _, ref := range ready {
		if ref.String() == lease.Task {
			return nil
		}
	}
	return errors.New("plan: claimed row is no longer eligible for this role")
}

// RequireDispatch rechecks the exact executable claim at admission and commit.
// Legacy callers retain current-row admission only while no dispatch claim owns
// that worktree or row; they cannot overwrite an active worker's authority.
func RequireDispatch(ctx context.Context, reader artifact.Reader, document Plan, authority CompletionAuthority, root, role, reference string) (Item, Step, *worklease.Lease, error) {
	mode, modeErr := ExecutionMode("")
	if modeErr != nil {
		return Item{}, Step{}, nil, modeErr
	}
	stop, stopErr := ReadStop(ctx, reader, root, "", mode)
	if stopErr != nil {
		return Item{}, Step{}, nil, stopErr
	}

	worker, err := dispatchWorker("")
	if err != nil {
		return Item{}, Step{}, nil, err
	}
	if err := processcontrol.RequireCampaign(root, worker); err != nil {
		return Item{}, Step{}, nil, err
	}
	if err := stop.RequireExecution(ctx, reader, role, worker, mode, reference, strings.TrimSpace(os.Getenv(AutomationMaintenanceEnvironment))); err != nil {
		return Item{}, Step{}, nil, err
	}
	worktree, err := filepath.Abs(root)
	if err != nil {
		return Item{}, Step{}, nil, err
	}
	worktree = filepath.ToSlash(worktree)
	lease, err := readDispatchAlias(ctx, reader, worklease.WorktreeAlias(worktree))
	if err != nil {
		return Item{}, Step{}, nil, err
	}
	rowLease, err := readDispatchAlias(ctx, reader, worklease.Alias(worklease.TaskAliasRoot, reference))
	if err != nil {
		return Item{}, Step{}, nil, err
	}
	if lease != nil && lease.Worker != "" || rowLease != nil || worker != "" {
		if lease == nil || rowLease == nil || worker == "" || lease.ID != rowLease.ID || lease.Worker != worker || lease.Task != reference {
			return Item{}, Step{}, nil, errors.New("plan: executable admission requires the exact worker's current worktree and row claim")
		}
		if err := worklease.ResolveOwner(ctx, reader, *lease); err != nil {
			return Item{}, Step{}, nil, err
		}
		if err := validateDispatchContract(document, *lease, role, authority); err != nil {
			return Item{}, Step{}, nil, err
		}
		item, step, _ := dispatchStep(document, reference)
		return item, step, lease, nil
	}
	item, step, found := Current(document, role, authority)
	if !found || item.ID+"/"+step.ID != reference {
		return Item{}, Step{}, nil, errors.New("plan: reference does not match the current open step")
	}
	return item, step, nil, nil
}

func selectDispatch(ctx context.Context, reader artifact.Reader, document Plan, role, worktree, worker, reference string, authority CompletionAuthority) (Dispatch, error) {
	aliases := []string{worklease.WorktreeAlias(worktree)}
	if worker != "" {
		aliases = append(aliases, worklease.Alias(worklease.WorkerAliasRoot, worker))
	}
	for _, alias := range aliases {
		lease, err := readDispatchAlias(ctx, reader, alias)
		if err != nil {
			return Dispatch{}, err
		}
		if lease == nil {
			continue
		}
		if reference != "" && lease.Task != reference {
			return finishDispatch(Dispatch{Claim: lease, Waiting: "release the existing claim before requesting another row"}), nil
		}
		if worker == "" || lease.Worker != worker || !strings.EqualFold(lease.Worktree, worktree) {
			return finishDispatch(Dispatch{Claim: lease, Waiting: fmt.Sprintf("%s owns %s in %s; inspect or use an isolated worktree", lease.Worker, lease.Task, lease.Worktree)}), nil
		}
		if err := worklease.ResolveOwner(ctx, reader, *lease); err != nil {
			return finishDispatch(Dispatch{Claim: lease, Waiting: err.Error()}), nil
		}
		if err := validateDispatchContract(document, *lease, role, authority); err != nil {
			return finishDispatch(Dispatch{Claim: lease, Waiting: err.Error()}), nil
		}
		item, step, _ := dispatchStep(document, lease.Task)
		return stepDispatch(item, step, lease), nil
	}
	ordered, err := dispatchPriority(document, role, authority)
	if err != nil {
		return Dispatch{}, err
	}
	var held *worklease.Lease
	for _, ref := range ordered {
		if reference != "" && ref.String() != reference {
			continue
		}
		lease, err := readDispatchAlias(ctx, reader, worklease.Alias(worklease.TaskAliasRoot, ref.String()))
		if err != nil {
			return Dispatch{}, err
		}
		if lease != nil {
			held = lease
			continue
		}
		item, step, _ := dispatchStep(document, ref.String())
		return stepDispatch(item, step, nil), nil
	}
	if held != nil {
		return finishDispatch(Dispatch{Claim: held, Waiting: "eligible rows are claimed; resume when an owner completes or hands off"}), nil
	}
	if reference != "" {
		return finishDispatch(Dispatch{Waiting: "requested row is not ready for this role"}), nil
	}
	if _, _, open := Current(document, role, authority); open {
		// Preserve the legacy empty-rung dispatch until it is given steps.
		return DispatchOf(document, role, authority), nil
	}
	for _, item := range document.Items {
		if item.Status == StatusOpen {
			return finishDispatch(Dispatch{Waiting: "remaining rows require prerequisite evidence or an eligible role"}), nil
		}
	}
	return finishDispatch(Dispatch{Complete: true}), nil
}
