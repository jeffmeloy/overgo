package operation

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/operatoraction"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

type State string

const (
	operationSequenceStep = 1
	operationEventBuffer  = 1

	StateAdmitted   State = "admitted"
	StatePreparing  State = "preparing"
	StateRunning    State = "running"
	StatePublishing State = "publishing"
	StateCompleted  State = "completed"
	StateBlocked    State = "blocked"
	StateCancelled  State = "cancelled"
	StateFailed     State = "failed"
)

type Progress struct {
	Completed uint64  `json:"completed"`
	Total     *uint64 `json:"total,omitempty"`
}

type Metric struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitzero"`
}

type Status struct {
	ID             artifact.ID              `json:"id"`
	Task           recipe.Task              `json:"task"`
	Recipe         artifact.ID              `json:"recipe"`
	State          State                    `json:"state"`
	Progress       Progress                 `json:"progress"`
	Metrics        []Metric                 `json:"metrics,omitempty"`
	Outputs        []artifact.ID            `json:"outputs,omitempty"`
	Attempts       []artifact.ID            `json:"attempts,omitempty"`
	Run            *artifact.ID             `json:"run,omitempty"`
	Failure        string                   `json:"failure,omitzero"`
	Recovery       *operatoraction.Block    `json:"recovery,omitempty"`
	WorkspaceClaim *WorkspaceClaimLifecycle `json:"workspace_claim,omitempty"`
	Deadline       *DeadlineState           `json:"deadline,omitempty"`
}

type Request struct {
	Task   recipe.Task
	Recipe artifact.ID
	Lease  *plan.WorkLease
	Effect *agenttool.InvocationEffect
	// Deadlines: the two-phase bounds; nil leaves the operation unbounded.
	Deadlines *Deadlines
}

// WorkspaceClaimState names one stage of a workspace claim lifecycle.
type WorkspaceClaimState string

const (
	// WorkspaceClaimAcquired marks a claim admitted for a fresh operation.
	WorkspaceClaimAcquired WorkspaceClaimState = "acquired"
	// WorkspaceClaimRecovered marks a claim re-admitted during recovery.
	WorkspaceClaimRecovered WorkspaceClaimState = "recovered"
	// WorkspaceClaimReleased marks a claim released when the operation ends.
	WorkspaceClaimReleased WorkspaceClaimState = "released"
)

// WorkspaceClaimLifecycle is immutable identity evidence retained in status.
type WorkspaceClaimLifecycle struct {
	ID       artifact.ID         `json:"-"`
	Lease    artifact.ID         `json:"lease"`
	Effect   artifact.ID         `json:"effect"`
	State    WorkspaceClaimState `json:"state"`
	Previous *artifact.ID        `json:"previous,omitempty"`
}

type Completion struct {
	Run     artifact.ID
	Outputs []artifact.ID
}

// Event carries one ordered operation snapshot.
type Event struct {
	Sequence uint64 `json:"sequence"`
	Status   Status `json:"status"`
}

// Reporter: operation identity and bounded progress publication.
type Reporter interface {
	OperationID() artifact.ID
	Progress(uint64, *uint64)
	Metric(Metric)
	Attempt(artifact.ID)
	Publishing()
}

type Executor func(context.Context, Reporter) (Completion, error)

type Manager struct {
	mu         sync.RWMutex
	entries    map[artifact.ID]*entry
	order      []artifact.ID
	limit      int
	salt       [sha256.Size]byte
	sequence   atomic.Uint64
	wait       sync.WaitGroup
	closed     bool
	watchers   map[uint64]chan Event
	watchID    uint64
	eventID    uint64
	repository artifact.Reader
	// now: the clock deadlines read; tests substitute a settable one.
	now  func() time.Time
	stop chan struct{}
}

type entry struct {
	status   Status
	request  Request
	execute  Executor
	cancel   context.CancelFunc
	done     chan struct{}
	deadline *deadlineTimer
}

type ticket struct {
	Salt     [sha256.Size]byte `json:"salt"`
	Sequence uint64            `json:"sequence"`
	Task     recipe.Task       `json:"task"`
	Recipe   artifact.ID       `json:"recipe"`
}

func NewManager(limit int) (*Manager, error) {
	return newManager(limit, nil)
}

// NewManagerWithRepository enables CAS-backed workspace claim admission.
func NewManagerWithRepository(limit int, repository artifact.Reader) (*Manager, error) {
	if repository == nil {
		return nil, errors.New("operation: workspace claim repository is absent")
	}
	return newManager(limit, repository)
}

func newManager(limit int, repository artifact.Reader) (*Manager, error) {
	if limit <= 0 {
		return nil, errors.New("operation: retention limit must be positive")
	}
	manager := &Manager{
		entries: make(map[artifact.ID]*entry), limit: limit, watchers: make(map[uint64]chan Event), repository: repository,
		now: time.Now, stop: make(chan struct{}),
	}
	if _, err := rand.Read(manager.salt[:]); err != nil {
		return nil, fmt.Errorf("operation: initialize identity: %w", err)
	}
	go manager.deadlineLoop(deadlineFlushInterval)
	return manager, nil
}

func (manager *Manager) Submit(parent context.Context, request Request, execute Executor) (artifact.ID, error) {
	if manager == nil || parent == nil || execute == nil {
		return artifact.ID{}, errors.New("operation: nil manager, context, or executor")
	}
	if !request.Task.Valid() || request.Recipe.Kind() != artifact.KindRecipe {
		return artifact.ID{}, errors.New("operation: invalid task or recipe")
	}
	id, err := artifact.JSONID(artifact.KindEvidence, ticket{
		Salt: manager.salt, Sequence: manager.sequence.Add(operationSequenceStep),
		Task: request.Task, Recipe: request.Recipe,
	})
	if err != nil {
		return artifact.ID{}, err
	}
	return manager.start(parent, id, request, execute, false)
}

// Recover resumes a terminal or process-lost operation identity.
func (manager *Manager) Recover(parent context.Context, id artifact.ID, request Request, execute Executor) (artifact.ID, error) {
	if id.Kind() != artifact.KindEvidence {
		return artifact.ID{}, errors.New("operation: invalid recovery identity")
	}
	return manager.start(parent, id, request, execute, true)
}

func (manager *Manager) start(parent context.Context, id artifact.ID, request Request, execute Executor, recover bool) (artifact.ID, error) {
	if manager == nil || parent == nil || execute == nil {
		return artifact.ID{}, errors.New("operation: nil manager, context, or executor")
	}
	if !request.Task.Valid() || request.Recipe.Kind() != artifact.KindRecipe {
		return artifact.ID{}, errors.New("operation: invalid task or recipe")
	}
	var timer *deadlineTimer
	if request.Deadlines != nil {
		if err := request.Deadlines.Validate(); err != nil {
			return artifact.ID{}, err
		}
		timer = &deadlineTimer{deadlines: *request.Deadlines, state: *newDeadlineState(*request.Deadlines, manager.now())}
	}
	claim, claimErr := manager.admitWorkspaceClaim(parent, request, recover)
	if claimErr != nil {
		return artifact.ID{}, claimErr
	}
	ctx, cancel := context.WithCancel(parent)
	manager.mu.Lock()
	if manager.closed {
		manager.mu.Unlock()
		cancel()
		return artifact.ID{}, errors.New("operation: manager is closed")
	}
	if claim != nil {
		for _, active := range manager.entries {
			if active.status.WorkspaceClaim != nil && active.status.WorkspaceClaim.State != WorkspaceClaimReleased &&
				active.request.Lease != nil && plan.WorkspaceClaimsConflict(*request.Lease, *active.request.Lease) {
				manager.mu.Unlock()
				cancel()
				return artifact.ID{}, errors.New("operation: workspace claim conflicts with active mutation")
			}
		}
	}
	if current := manager.entries[id]; current != nil {
		if !recover || !terminal(current.status.State) {
			manager.mu.Unlock()
			cancel()
			return artifact.ID{}, errors.New("operation: operation is active or already admitted")
		}
		if current.status.Task != request.Task || current.status.Recipe != request.Recipe {
			manager.mu.Unlock()
			cancel()
			return artifact.ID{}, ErrLifecycleConflict
		}
	}
	manager.entries[id] = &entry{
		status: Status{
			ID: id, Task: request.Task, Recipe: request.Recipe, State: StateAdmitted, WorkspaceClaim: claim,
			Deadline: deadlineStatus(timer),
		},
		request:  request,
		execute:  execute,
		cancel:   cancel,
		done:     make(chan struct{}),
		deadline: timer,
	}
	if !slices.Contains(manager.order, id) {
		manager.order = append(manager.order, id)
	}
	manager.publishLocked(manager.entries[id].status)
	manager.trimLocked()
	manager.mu.Unlock()
	manager.wait.Go(func() { manager.run(ctx, id, execute) })
	return id, nil
}

func (manager *Manager) run(ctx context.Context, id artifact.ID, execute Executor) {
	defer func() {
		manager.mu.Lock()
		if current := manager.entries[id]; current != nil {
			manager.releaseWorkspaceClaim(current)
			current.cancel()
			current.cancel = nil
			if current.status.State != StateBlocked {
				current.execute = nil
			}
			close(current.done)
			manager.trimLocked()
			manager.publishLocked(current.status)
		}
		manager.mu.Unlock()
	}()
	reporter := operationReporter{manager: manager, id: id}
	manager.setState(id, StatePreparing)
	manager.setState(id, StateRunning)
	completion, err := execute(ctx, reporter)
	manager.mu.Lock()
	defer manager.mu.Unlock()
	current := manager.entries[id]
	if current == nil {
		return
	}
	completionErr := validateCompletion(completion)
	switch {
	case completionErr != nil:
		current.status.State = StateFailed
		current.status.Failure = completionErr.Error()
	case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil:
		current.status.Run = &completion.Run
		current.status.State = StateCancelled
	case err != nil:
		current.status.Run = &completion.Run
		if recovery, ok := operatoraction.Recovery(err); ok {
			if recovery.Subject == current.status.Recipe {
				current.status.State = StateBlocked
				recovery = recovery.Clone()
				current.status.Recovery = &recovery
			} else {
				current.status.State = StateFailed
			}
		} else {
			current.status.State = StateFailed
		}
		current.status.Failure = err.Error()
	default:
		current.status.Run = &completion.Run
		current.status.Outputs = slices.Clone(completion.Outputs)
		current.status.State = StateCompleted
	}
	manager.publishLocked(current.status)
}

func (manager *Manager) admitWorkspaceClaim(ctx context.Context, request Request, recover bool) (*WorkspaceClaimLifecycle, error) {
	if request.Effect == nil || request.Effect.Class != agenttool.EffectMutation {
		return nil, nil
	}
	if request.Lease == nil || manager.repository == nil || !request.Effect.ID.Valid() {
		return nil, errors.New("operation: mutation requires exact effect, work lease, and claim repository")
	}
	if err := plan.ResolveWorkLeaseOwner(ctx, manager.repository, *request.Lease); err != nil {
		return nil, err
	}
	if request.Effect.OpaqueMutation || !request.Effect.Known {
		return nil, errors.New("operation: opaque mutation cannot acquire a scoped workspace claim")
	}
	state := WorkspaceClaimAcquired
	expires, err := time.Parse(time.RFC3339Nano, request.Lease.ExpiresAt)
	if err != nil {
		return nil, err
	}
	if !expires.After(time.Now()) {
		if !recover {
			return nil, errors.New("operation: work lease expired before admission")
		}
		state = WorkspaceClaimRecovered
	}
	claim := WorkspaceClaimLifecycle{Lease: request.Lease.ID, Effect: request.Effect.ID, State: state}
	claim.ID, err = artifact.JSONID(artifact.KindEvidence, claim)
	return &claim, err
}

func (manager *Manager) releaseWorkspaceClaim(current *entry) {
	if current == nil || current.status.WorkspaceClaim == nil || current.status.WorkspaceClaim.State == WorkspaceClaimReleased {
		return
	}
	prior := current.status.WorkspaceClaim
	released := WorkspaceClaimLifecycle{Lease: prior.Lease, Effect: prior.Effect, State: WorkspaceClaimReleased, Previous: artifact.IDPointer(prior.ID)}
	released.ID, _ = artifact.JSONID(artifact.KindEvidence, released)
	current.status.WorkspaceClaim = &released
}

// RecoverAfterDecision resumes only the exact action bound by a granted decision.
func (manager *Manager) RecoverAfterDecision(
	parent context.Context,
	decision runrecord.HumanDecision,
) (artifact.ID, error) {
	if manager == nil || parent == nil || decision.Answer != operatoraction.AnswerGrant {
		return artifact.ID{}, errors.New("operation: granted recovery decision required")
	}
	manager.mu.RLock()
	current := manager.entries[decision.Operation]
	if current == nil || current.status.State != StateBlocked || current.status.Recovery == nil ||
		current.request.Recipe != decision.Recipe || current.execute == nil {
		manager.mu.RUnlock()
		return artifact.ID{}, errors.New("operation: blocked recovery is unavailable")
	}
	admitted := false
	for _, action := range current.status.Recovery.Actions {
		if action.Code == decision.Tool && slices.Equal(action.Argv, decision.Arguments) {
			admitted = true
			break
		}
	}
	request, execute := current.request, current.execute
	manager.mu.RUnlock()
	if !admitted {
		return artifact.ID{}, errors.New("operation: decision action differs from recovery contract")
	}
	return manager.Recover(parent, decision.Operation, request, execute)
}

func validateCompletion(completion Completion) error {
	if completion.Run.Kind() != artifact.KindRun {
		return errors.New("operation: completion lacks durable run")
	}
	for _, output := range completion.Outputs {
		if !output.Valid() {
			return errors.New("operation: completion has invalid output")
		}
	}
	return nil
}

func (manager *Manager) Status(id artifact.ID) (Status, bool) {
	if manager == nil {
		return Status{}, false
	}
	manager.mu.RLock()
	defer manager.mu.RUnlock()
	current, ok := manager.entries[id]
	if !ok {
		return Status{}, false
	}
	return cloneStatus(current.status), true
}

func (manager *Manager) List() []Status {
	if manager == nil {
		return nil
	}
	manager.mu.RLock()
	defer manager.mu.RUnlock()
	result := make([]Status, 0, len(manager.entries))
	for _, id := range manager.order {
		if current := manager.entries[id]; current != nil {
			result = append(result, cloneStatus(current.status))
		}
	}
	return result
}

// Subscribe returns latest-only operation events.
func (manager *Manager) Subscribe() (<-chan Event, func(), error) {
	if manager == nil {
		return nil, nil, errors.New("operation: nil manager")
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.closed || len(manager.watchers) >= manager.limit {
		return nil, nil, errors.New("operation: event subscription unavailable")
	}
	manager.watchID++
	id := manager.watchID
	events := make(chan Event, operationEventBuffer)
	manager.watchers[id] = events
	return events, func() {
		manager.mu.Lock()
		if current := manager.watchers[id]; current != nil {
			delete(manager.watchers, id)
			close(current)
		}
		manager.mu.Unlock()
	}, nil
}

func (manager *Manager) Wait(ctx context.Context, id artifact.ID) (Status, error) {
	if manager == nil || ctx == nil {
		return Status{}, errors.New("operation: nil manager or context")
	}
	manager.mu.RLock()
	current := manager.entries[id]
	if current == nil {
		manager.mu.RUnlock()
		return Status{}, errors.New("operation: unknown operation")
	}
	done := current.done
	manager.mu.RUnlock()
	select {
	case <-ctx.Done():
		return Status{}, ctx.Err()
	case <-done:
		status, _ := manager.Status(id)
		return status, nil
	}
}

func (manager *Manager) Cancel(id artifact.ID) bool {
	if manager == nil {
		return false
	}
	manager.mu.RLock()
	current := manager.entries[id]
	if current == nil || current.cancel == nil || terminal(current.status.State) {
		manager.mu.RUnlock()
		return false
	}
	cancel := current.cancel
	manager.mu.RUnlock()
	cancel()
	return true
}

func (manager *Manager) Close() {
	if manager == nil {
		return
	}
	manager.mu.Lock()
	if !manager.closed {
		close(manager.stop)
	}
	manager.closed = true
	for _, current := range manager.entries {
		if current.cancel != nil && !terminal(current.status.State) {
			current.cancel()
		}
	}
	for id, watcher := range manager.watchers {
		delete(manager.watchers, id)
		close(watcher)
	}
	manager.mu.Unlock()
	manager.wait.Wait()
}

func (manager *Manager) setState(id artifact.ID, state State) {
	manager.mu.Lock()
	if current := manager.entries[id]; current != nil && !terminal(current.status.State) {
		current.status.State = state
		manager.publishLocked(current.status)
	}
	manager.mu.Unlock()
}

func (manager *Manager) publishLocked(status Status) {
	if len(manager.watchers) == 0 {
		return
	}
	manager.eventID++
	for _, target := range manager.watchers {
		event := Event{Sequence: manager.eventID, Status: cloneStatus(status)}
		select {
		case target <- event:
		default:
			select {
			case <-target:
			default:
			}
			select {
			case target <- event:
			default:
			}
		}
	}
}

func (manager *Manager) trimLocked() {
	for len(manager.entries) > manager.limit {
		removed := false
		for index, id := range manager.order {
			if current := manager.entries[id]; current != nil && terminal(current.status.State) {
				delete(manager.entries, id)
				manager.order = append(manager.order[:index], manager.order[index+1:]...)
				removed = true
				break
			}
		}
		if !removed {
			return
		}
	}
}

func terminal(state State) bool {
	return state == StateCompleted || state == StateBlocked || state == StateCancelled || state == StateFailed
}

func cloneStatus(status Status) Status {
	status.Metrics = slices.Clone(status.Metrics)
	status.Outputs = slices.Clone(status.Outputs)
	status.Attempts = slices.Clone(status.Attempts)
	if status.Run != nil {
		run := *status.Run
		status.Run = &run
	}
	if status.Progress.Total != nil {
		total := *status.Progress.Total
		status.Progress.Total = &total
	}
	if status.Recovery != nil {
		recovery := status.Recovery.Clone()
		status.Recovery = &recovery
	}
	if status.WorkspaceClaim != nil {
		claim := *status.WorkspaceClaim
		status.WorkspaceClaim = &claim
	}
	if status.Deadline != nil {
		deadline := *status.Deadline
		status.Deadline = &deadline
	}
	return status
}

type operationReporter struct {
	manager *Manager
	id      artifact.ID
}

func (reporter operationReporter) OperationID() artifact.ID { return reporter.id }

// Attempt records ordered serving evidence.
func (reporter operationReporter) Attempt(id artifact.ID) {
	if id.Kind() != artifact.KindEvidence {
		return
	}
	reporter.manager.mu.Lock()
	if current := reporter.manager.entries[reporter.id]; current != nil &&
		!terminal(current.status.State) && !slices.Contains(current.status.Attempts, id) {
		current.status.Attempts = append(current.status.Attempts, id)
		if len(current.status.Attempts) > reporter.manager.limit {
			current.status.Attempts = slices.Clone(current.status.Attempts[len(current.status.Attempts)-reporter.manager.limit:])
		}
		reporter.manager.publishLocked(current.status)
	}
	reporter.manager.mu.Unlock()
}

func (reporter operationReporter) Progress(completed uint64, total *uint64) {
	reporter.manager.mu.Lock()
	defer reporter.manager.mu.Unlock()
	if current := reporter.manager.entries[reporter.id]; current != nil && !terminal(current.status.State) {
		current.status.Progress.Completed = completed
		current.status.Progress.Total = nil
		if total != nil {
			value := *total
			current.status.Progress.Total = &value
		}
		// A report starts or refreshes the execution deadline.
		if current.deadline != nil {
			current.deadline.report(reporter.manager.now())
			current.status.Deadline = deadlineStatus(current.deadline)
		}
		reporter.manager.publishLocked(current.status)
	}
}

func (reporter operationReporter) Metric(metric Metric) {
	if metric.Name == "" || !checked.Finite64(metric.Value) {
		return
	}
	reporter.manager.mu.Lock()
	defer reporter.manager.mu.Unlock()
	if current := reporter.manager.entries[reporter.id]; current != nil && !terminal(current.status.State) {
		for index := range current.status.Metrics {
			if current.status.Metrics[index].Name == metric.Name {
				current.status.Metrics[index] = metric
				reporter.manager.publishLocked(current.status)
				return
			}
		}
		current.status.Metrics = append(current.status.Metrics, metric)
		if len(current.status.Metrics) > reporter.manager.limit {
			current.status.Metrics = slices.Clone(current.status.Metrics[len(current.status.Metrics)-reporter.manager.limit:])
		}
		reporter.manager.publishLocked(current.status)
	}
}

func (reporter operationReporter) Publishing() {
	reporter.manager.setState(reporter.id, StatePublishing)
}
