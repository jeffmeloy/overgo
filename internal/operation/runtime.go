package operation

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"sync/atomic"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

type State string

const (
	StateAdmitted   State = "admitted"
	StatePreparing  State = "preparing"
	StateRunning    State = "running"
	StatePublishing State = "publishing"
	StateCompleted  State = "completed"
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
	Unit  string  `json:"unit,omitempty"`
}

type Status struct {
	ID       artifact.ID   `json:"id"`
	Task     recipe.Task   `json:"task"`
	Recipe   artifact.ID   `json:"recipe"`
	State    State         `json:"state"`
	Progress Progress      `json:"progress"`
	Metrics  []Metric      `json:"metrics,omitempty"`
	Outputs  []artifact.ID `json:"outputs,omitempty"`
	Run      *artifact.ID  `json:"run,omitempty"`
	Failure  string        `json:"failure,omitempty"`
}

type Request struct {
	Task   recipe.Task
	Recipe artifact.ID
}

type Completion struct {
	Run     artifact.ID
	Outputs []artifact.ID
}

type Reporter interface {
	Progress(uint64, *uint64)
	Metric(Metric)
	Publishing()
}

type Executor func(context.Context, Reporter) (Completion, error)

type Manager struct {
	mu       sync.RWMutex
	entries  map[artifact.ID]*entry
	order    []artifact.ID
	limit    int
	salt     [sha256.Size]byte
	sequence atomic.Uint64
	wait     sync.WaitGroup
	closed   bool
}

type entry struct {
	status Status
	cancel context.CancelFunc
	done   chan struct{}
}

type ticket struct {
	Salt     [sha256.Size]byte `json:"salt"`
	Sequence uint64            `json:"sequence"`
	Task     recipe.Task       `json:"task"`
	Recipe   artifact.ID       `json:"recipe"`
}

func NewManager(limit int) (*Manager, error) {
	if limit <= 0 {
		return nil, errors.New("operation: retention limit must be positive")
	}
	manager := &Manager{entries: make(map[artifact.ID]*entry), limit: limit}
	if _, err := rand.Read(manager.salt[:]); err != nil {
		return nil, fmt.Errorf("operation: initialize identity: %w", err)
	}
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
		Salt: manager.salt, Sequence: manager.sequence.Add(1),
		Task: request.Task, Recipe: request.Recipe,
	})
	if err != nil {
		return artifact.ID{}, err
	}
	ctx, cancel := context.WithCancel(parent)
	manager.mu.Lock()
	if manager.closed {
		manager.mu.Unlock()
		cancel()
		return artifact.ID{}, errors.New("operation: manager is closed")
	}
	manager.entries[id] = &entry{
		status: Status{ID: id, Task: request.Task, Recipe: request.Recipe, State: StateAdmitted},
		cancel: cancel,
		done:   make(chan struct{}),
	}
	manager.order = append(manager.order, id)
	manager.trimLocked()
	manager.wait.Add(1)
	manager.mu.Unlock()
	go manager.run(ctx, id, execute)
	return id, nil
}

func (manager *Manager) run(ctx context.Context, id artifact.ID, execute Executor) {
	defer manager.wait.Done()
	defer func() {
		manager.mu.Lock()
		if current := manager.entries[id]; current != nil {
			current.cancel()
			current.cancel = nil
			close(current.done)
			manager.trimLocked()
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
		current.status.State = StateFailed
		current.status.Failure = err.Error()
	default:
		current.status.Run = &completion.Run
		current.status.Outputs = slices.Clone(completion.Outputs)
		current.status.State = StateCompleted
	}
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
	manager.closed = true
	for _, current := range manager.entries {
		if current.cancel != nil && !terminal(current.status.State) {
			current.cancel()
		}
	}
	manager.mu.Unlock()
	manager.wait.Wait()
}

func (manager *Manager) setState(id artifact.ID, state State) {
	manager.mu.Lock()
	if current := manager.entries[id]; current != nil && !terminal(current.status.State) {
		current.status.State = state
	}
	manager.mu.Unlock()
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
	return state == StateCompleted || state == StateCancelled || state == StateFailed
}

func cloneStatus(status Status) Status {
	status.Metrics = slices.Clone(status.Metrics)
	status.Outputs = slices.Clone(status.Outputs)
	if status.Run != nil {
		run := *status.Run
		status.Run = &run
	}
	if status.Progress.Total != nil {
		total := *status.Progress.Total
		status.Progress.Total = &total
	}
	return status
}

type operationReporter struct {
	manager *Manager
	id      artifact.ID
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
	}
}

func (reporter operationReporter) Metric(metric Metric) {
	if metric.Name == "" || math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) {
		return
	}
	reporter.manager.mu.Lock()
	defer reporter.manager.mu.Unlock()
	if current := reporter.manager.entries[reporter.id]; current != nil && !terminal(current.status.State) {
		current.status.Metrics = append(current.status.Metrics, metric)
	}
}

func (reporter operationReporter) Publishing() {
	reporter.manager.setState(reporter.id, StatePublishing)
}
