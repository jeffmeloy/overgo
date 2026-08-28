package capabilityruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/jsonabbrev"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

var (
	ErrAdmissionQueueFull = errors.New("capability runtime: admission queue is full")
	ErrSessionUnavailable = errors.New("capability runtime: model session is unavailable")
)

type sessionKey struct {
	model, recipe, resources artifact.ID
	device, policy           string
}

// SessionInput identifies request state that controls resident compatibility.
type SessionInput interface {
	SessionKey() (string, error)
}

type sessionEntry[Model any] struct {
	mu        sync.Mutex
	model     Model
	scope     artifact.ID
	ready     chan struct{}
	loadErr   error
	used      uint64
	borrowers int
	dead      bool
	closing   bool
	retiring  bool
	closed    bool
}

// ModelSessionDirector: bounded recipe-bound model sessions.
type ModelSessionDirector[Input, Model, Output any] struct {
	name     string
	device   string
	capacity int
	validate func(Input) error
	policy   func(Input) (string, error)
	load     func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error)
	reset    func(context.Context, Model, Input) error
	bind     func(*workflowruntime.Runtime, artifact.ID, Model) error
	inputs   func(Input, artifact.Content, recipe.Definition) (map[recipe.PortName]workflowruntime.Value, error)

	mu      sync.Mutex
	tick    uint64
	entries map[sessionKey]*sessionEntry[Model]
	changed chan struct{}
	closed  bool

	admissions    chan int
	admissionMu   sync.Mutex
	resident      Model
	residentSet   bool
	residentUsers int
	residentClose bool
	waiters       int
	loads         uint64
	reuses        uint64
	evictions     uint64
	retirements   uint64
	retiring      int
	retiringUsers int
}

// SessionLease: admitted use of one resident model session.
type SessionLease[Model any] struct {
	ID      int
	model   Model
	release func() error
	once    sync.Once
	err     error
}

// SessionSnapshot: current resident admission state.
type SessionSnapshot struct {
	Name        string `json:"name"`
	Device      string `json:"device"`
	Capacity    int    `json:"capacity"`
	Active      int    `json:"active"`
	Waiting     int    `json:"waiting"`
	Available   int    `json:"available"`
	Entries     int    `json:"entries"`
	Idle        int    `json:"idle"`
	Loads       uint64 `json:"loads"`
	Reuses      uint64 `json:"reuses"`
	Evictions   uint64 `json:"evictions"`
	Retirements uint64 `json:"retirements"`
	Retiring    int    `json:"retiring"`
	Closed      bool   `json:"closed"`
}

func (l *SessionLease[Model]) Model() Model { return l.model }

func (l *SessionLease[Model]) Release() error {
	if l != nil && l.release != nil {
		l.once.Do(func() { l.err = l.release() })
	}
	return l.err
}

// NewResidentModelSessionDirector adopts one compiled serving session.
func NewResidentModelSessionDirector[Model any](
	name, device string,
	capacity int,
	model Model,
) (*ModelSessionDirector[struct{}, Model, struct{}], error) {
	if name == "" || device == "" || capacity <= 0 {
		return nil, errors.New("capability runtime: incomplete resident model session director")
	}
	admissions := make(chan int, capacity)
	for id := range capacity {
		admissions <- id
	}
	return &ModelSessionDirector[struct{}, Model, struct{}]{
		name: name, device: device, capacity: capacity,
		entries: make(map[sessionKey]*sessionEntry[Model]), changed: make(chan struct{}),
		admissions: admissions, resident: model, residentSet: true,
	}, nil
}

// TryLease admits one request without waiting.
func (c *ModelSessionDirector[Input, Model, Output]) TryLease(requested int) (*SessionLease[Model], bool) {
	if c == nil || c.admissions == nil {
		return nil, false
	}
	c.admissionMu.Lock()
	defer c.admissionMu.Unlock()
	id := -1
	if requested < 0 {
		select {
		case id = <-c.admissions:
		default:
			return nil, false
		}
	} else {
		available := len(c.admissions)
		for range available {
			candidate := <-c.admissions
			if candidate == requested {
				id = candidate
			} else {
				c.admissions <- candidate
			}
			if id >= 0 {
				break
			}
		}
		if id < 0 {
			return nil, false
		}
	}
	c.mu.Lock()
	if c.closed || !c.residentSet {
		c.mu.Unlock()
		c.admissions <- id
		return nil, false
	}
	c.residentUsers++
	model := c.resident
	c.mu.Unlock()
	lease := &SessionLease[Model]{ID: id, model: model}
	lease.release = func() error {
		c.admissionMu.Lock()
		c.admissions <- id
		c.admissionMu.Unlock()
		c.mu.Lock()
		c.residentUsers--
		c.notify()
		c.mu.Unlock()
		return nil
	}
	return lease, true
}

// Lease parks behind resident capacity while the request context remains
// live. At most one waiter per configured slot is admitted to the parking
// set; additional callers fail without joining an unbounded queue.
func (c *ModelSessionDirector[Input, Model, Output]) Lease(ctx context.Context, requested int) (*SessionLease[Model], error) {
	if c == nil || c.admissions == nil || ctx == nil {
		return nil, ErrSessionUnavailable
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if requested >= c.capacity {
		return nil, ErrSessionUnavailable
	}
	if lease, ok := c.TryLease(requested); ok {
		return lease, nil
	}
	c.mu.Lock()
	if c.closed || !c.residentSet {
		c.mu.Unlock()
		return nil, ErrSessionUnavailable
	}
	if c.waiters >= c.capacity {
		c.mu.Unlock()
		return nil, ErrAdmissionQueueFull
	}
	c.waiters++
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		c.waiters--
		c.mu.Unlock()
	}()
	for {
		c.mu.Lock()
		if c.closed || !c.residentSet {
			c.mu.Unlock()
			return nil, ErrSessionUnavailable
		}
		changed := c.changed
		c.mu.Unlock()
		if lease, ok := c.TryLease(requested); ok {
			return lease, nil
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (c *ModelSessionDirector[Input, Model, Output]) Available() int {
	if c == nil || c.admissions == nil {
		return 0
	}
	return len(c.admissions)
}

func (c *ModelSessionDirector[Input, Model, Output]) Snapshot() SessionSnapshot {
	if c == nil {
		return SessionSnapshot{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	available := 0
	if c.admissions != nil {
		available = len(c.admissions)
	}
	var noBorrowers int
	active, idle, busy := c.residentUsers+c.retiringUsers, noBorrowers, noBorrowers
	for _, entry := range c.entries {
		active += entry.borrowers
		if entry.borrowers == noBorrowers {
			idle++
		} else {
			busy++
		}
	}
	if c.admissions == nil {
		available = c.capacity - busy - c.retiring
	}
	return SessionSnapshot{
		Name: c.name, Device: c.device, Capacity: c.capacity,
		Active: active, Waiting: c.waiters, Available: available,
		Entries: len(c.entries), Idle: idle,
		Loads: c.loads, Reuses: c.reuses, Evictions: c.evictions,
		Retirements: c.retirements, Retiring: c.retiring,
		Closed: c.closed,
	}
}

func NewModelSessionDirector[Input SessionInput, Model, Output any](
	name, device string,
	capacity int,
	validate func(Input) error,
	load func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error),
	reset func(context.Context, Model, Input) error,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) (*ModelSessionDirector[Input, Model, Output], error) {
	return newModelSessionDirector[Input, Model, Output](
		name, device, capacity, validate, func(input Input) (string, error) { return input.SessionKey() }, load, reset, bind,
		func(input Input, content artifact.Content, definition recipe.Definition) (map[recipe.PortName]workflowruntime.Value, error) {
			if len(definition.Inputs) != 1 {
				return nil, errors.New("capability runtime: scalar program identity differs")
			}
			port := definition.Inputs[0]
			return map[recipe.PortName]workflowruntime.Value{
				port.Name: workflowruntime.ArtifactValue(port.Data, input, content),
			}, nil
		},
	)
}

// NewComponentSessionDirector: recipe-component lease owner.
func NewComponentSessionDirector[Model any](name, device string, capacity int) (*ModelSessionDirector[struct{}, Model, struct{}], error) {
	if name == "" || device == "" || capacity <= 0 {
		return nil, errors.New("capability runtime: incomplete component session director")
	}
	return &ModelSessionDirector[struct{}, Model, struct{}]{
		name: name, device: device, capacity: capacity,
		entries: make(map[sessionKey]*sessionEntry[Model]), changed: make(chan struct{}),
	}, nil
}

// LeaseComponent: exclusive use under its compiled lifetime.
func (c *ModelSessionDirector[Input, Model, Output]) LeaseComponent(
	ctx context.Context,
	component modelrecipe.ComponentSession,
	load func(context.Context) (Model, error),
) (*SessionLease[Model], error) {
	if c == nil || ctx == nil || load == nil || !component.Identity.Valid() || !component.Model.Valid() || component.Session == "" {
		return nil, errors.New("capability runtime: incomplete component lease")
	}
	key := sessionKey{
		model: component.Model, resources: component.Identity,
		device: c.device, policy: string(component.Session),
	}
	entry, _, err := c.leaseEntry(ctx, key, component.Identity, nil, load)
	if err != nil {
		return nil, err
	}
	lease := &SessionLease[Model]{model: entry.model}
	lease.release = func() error {
		var closeErr error
		if component.Session == recipe.SessionRequest {
			closeErr = c.retire(ctx, key, entry, nil)
		}
		c.release(entry)
		return closeErr
	}
	return lease, nil
}

// MappedInput binds one JSON request field to a typed recipe port.
type MappedInput struct {
	Value   any
	Content artifact.Content
}

// NewMappedModelSessionDirector: multi-input session authority.
func NewMappedModelSessionDirector[Input SessionInput, Model, Output any](
	name, device string,
	capacity int,
	validate func(Input) error,
	load func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error),
	reset func(context.Context, Model, Input) error,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
	mapInputs func(Input) (map[recipe.PortName]MappedInput, error),
) (*ModelSessionDirector[Input, Model, Output], error) {
	if mapInputs == nil {
		return nil, errors.New("capability runtime: input mapper is nil")
	}
	return newModelSessionDirector[Input, Model, Output](
		name, device, capacity, validate, func(input Input) (string, error) { return input.SessionKey() }, load, reset, bind,
		func(input Input, _ artifact.Content, definition recipe.Definition) (map[recipe.PortName]workflowruntime.Value, error) {
			mapped, err := mapInputs(input)
			if err != nil {
				return nil, err
			}
			if len(mapped) != len(definition.Inputs) {
				return nil, errors.New("capability runtime: mapped input set differs")
			}
			values := make(map[recipe.PortName]workflowruntime.Value, len(mapped))
			for _, port := range definition.Inputs {
				input, ok := mapped[port.Name]
				if !ok {
					return nil, fmt.Errorf("capability runtime: mapped input %q is absent", port.Name)
				}
				if err := input.Content.Validate(); err != nil {
					return nil, err
				}
				values[port.Name] = workflowruntime.ArtifactValue(port.Data, input.Value, input.Content)
			}
			return values, nil
		},
	)
}

func newModelSessionDirector[Input, Model, Output any](
	name, device string,
	capacity int,
	validate func(Input) error,
	policy func(Input) (string, error),
	load func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error),
	reset func(context.Context, Model, Input) error,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
	inputs func(Input, artifact.Content, recipe.Definition) (map[recipe.PortName]workflowruntime.Value, error),
) (*ModelSessionDirector[Input, Model, Output], error) {
	if name == "" || device == "" || capacity <= 0 || validate == nil || policy == nil || load == nil || reset == nil || bind == nil {
		return nil, errors.New("capability runtime: incomplete model session director")
	}
	return &ModelSessionDirector[Input, Model, Output]{
		name: name, device: device, capacity: capacity,
		validate: validate, policy: policy, load: load, reset: reset, bind: bind, inputs: inputs,
		entries: make(map[sessionKey]*sessionEntry[Model]), changed: make(chan struct{}),
	}, nil
}

func (c *ModelSessionDirector[Input, Model, Output]) Executor() Executor {
	return c.execute
}

func (c *ModelSessionDirector[Input, Model, Output]) execute(
	ctx context.Context,
	store artifact.Repository,
	path string,
	execution modelrecipe.CapabilityEvidenceSelection,
	raw string,
) (any, error) {
	var zero Output
	execution, err := modelrecipe.RefreshCapabilityExecution(ctx, store, execution)
	if err != nil {
		return zero, err
	}
	program := execution.Program
	modelID := program.Definition().Model
	input, content, err := decodeJSONInput(c.name, c.validate, modelID, program, raw)
	if err != nil {
		return zero, err
	}
	policy, err := c.policy(input)
	if err != nil || policy == "" {
		return zero, errors.Join(errors.New("capability runtime: invalid execution policy"), err)
	}
	definition, resources := program.Definition(), execution.Resources
	key := sessionKey{
		model: modelID, recipe: definition.ID, resources: resources.Identity,
		device: c.device, policy: policy,
	}

	entry, fresh, err := c.lease(ctx, store, path, key, execution, input)
	if err != nil {
		return zero, err
	}
	defer c.release(entry)
	if !fresh {
		if err := c.reset(ctx, entry.model, input); err != nil {
			return zero, c.retire(ctx, key, entry, err)
		}
	}
	inputs, err := c.inputs(input, content, program.Definition())
	if err == nil {
		output, walls, executeErr := ExecuteMeasured[Output](
			ctx, store, modelID, program,
			"recipe/run/"+program.Definition().ID.String()+"/"+content.Descriptor.ID.String(), inputs,
			func(runtime *workflowruntime.Runtime) error { return c.bind(runtime, modelID, entry.model) },
		)
		var measured Measured
		if executeErr == nil {
			measured = measure(output, entry.model, walls)
		}
		if executeErr != nil {
			err = c.retire(ctx, key, entry, executeErr)
		} else if resources.RequestScoped() {
			err = c.retire(ctx, key, entry, nil)
		}
		if executeErr != nil {
			return zero, err
		}
		return measured, err
	}
	if err != nil {
		err = c.retire(ctx, key, entry, err)
	}
	return zero, err
}

func (c *ModelSessionDirector[Input, Model, Output]) lease(
	ctx context.Context,
	store artifact.Repository,
	path string,
	key sessionKey,
	execution modelrecipe.CapabilityEvidenceSelection,
	input Input,
) (*sessionEntry[Model], bool, error) {
	return c.leaseEntry(ctx, key, execution.Scope,
		func(ctx context.Context) error {
			_, err := modelrecipe.RefreshCapabilityExecution(ctx, store, execution)
			return err
		},
		func(ctx context.Context) (Model, error) {
			return c.load(ctx, store, path, execution.Program, input)
		})
}

func (c *ModelSessionDirector[Input, Model, Output]) leaseEntry(
	ctx context.Context,
	key sessionKey,
	scope artifact.ID,
	refresh func(context.Context) error,
	load func(context.Context) (Model, error),
) (*sessionEntry[Model], bool, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return nil, false, errors.New("capability runtime: model session director is closed")
		}
		c.tick++
		retired := c.supersedeLocked(scope, key)
		if len(retired) > 0 {
			c.notify()
			c.mu.Unlock()
			for _, stale := range retired {
				c.finishRetirement(context.WithoutCancel(ctx), stale)
			}
			continue
		}
		if entry, ok := c.entries[key]; ok {
			entry.scope = scope
			entry.used = c.tick
			select {
			case <-entry.ready:
				if entry.loadErr != nil {
					loadErr := entry.loadErr
					c.mu.Unlock()
					return nil, false, loadErr
				}
				if entry.mu.TryLock() {
					if entry.dead {
						entry.mu.Unlock()
						c.mu.Unlock()
						continue
					}
					entry.borrowers++
					c.reuses++
					c.mu.Unlock()
					return entry, false, nil
				}
				changed := c.changed
				if err := c.parkLocked(ctx, changed); err != nil {
					return nil, false, err
				}
				continue
			default:
				if err := c.parkLocked(ctx, entry.ready); err != nil {
					return nil, false, err
				}
				if entry.loadErr != nil {
					return nil, false, entry.loadErr
				}
				continue
			}
		}
		if len(c.entries)+c.retiring >= c.capacity {
			oldestKey, oldest := c.oldestIdle()
			if oldest == nil {
				changed := c.changed
				if err := c.parkLocked(ctx, changed); err != nil {
					return nil, false, err
				}
				continue
			}
			oldest.dead = true
			delete(c.entries, oldestKey)
			c.evictions++
			c.notify()
			c.mu.Unlock()
			oldest.mu.Lock()
			err := closeEntry(context.WithoutCancel(ctx), oldest)
			oldest.mu.Unlock()
			if err != nil {
				return nil, false, err
			}
			continue
		}
		entry := &sessionEntry[Model]{
			ready: make(chan struct{}), scope: scope, used: c.tick, borrowers: 1,
		}
		c.entries[key] = entry
		c.notify()
		c.mu.Unlock()
		model, err := load(ctx)
		if err == nil && refresh != nil {
			err = refresh(ctx)
		}
		entry.mu.Lock()
		entry.model, entry.loadErr = model, err
		close(entry.ready)
		c.mu.Lock()
		if err == nil {
			c.loads++
		} else {
			entry.dead = true
			if c.entries[key] == entry {
				delete(c.entries, key)
				entry.retiring = true
				c.retiring++
				c.retiringUsers += entry.borrowers
			}
		}
		dead := entry.dead
		c.notify()
		c.mu.Unlock()
		if err != nil {
			entry.mu.Unlock()
			c.releaseBorrower(entry)
			return nil, false, err
		}
		if dead {
			entry.mu.Unlock()
			c.releaseBorrower(entry)
			return nil, false, errors.New("capability runtime: model session director is closed")
		}
		return entry, true, nil
	}
}

// parkLocked joins the bounded, context-aware wait set and releases c.mu.
func (c *ModelSessionDirector[Input, Model, Output]) parkLocked(ctx context.Context, signal <-chan struct{}) error {
	if c.waiters >= c.capacity {
		c.mu.Unlock()
		return ErrAdmissionQueueFull
	}
	c.waiters++
	c.mu.Unlock()
	var err error
	select {
	case <-signal:
	case <-ctx.Done():
		err = ctx.Err()
	}
	c.mu.Lock()
	c.waiters--
	c.notify()
	c.mu.Unlock()
	return err
}

func (c *ModelSessionDirector[Input, Model, Output]) Close(ctx context.Context) error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.notify()
	for c.residentUsers > 0 || c.retiring > 0 {
		changed := c.changed
		c.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
		c.mu.Lock()
	}
	entries := make([]*sessionEntry[Model], 0, len(c.entries))
	for key, entry := range c.entries {
		entry.dead = true
		entries = append(entries, entry)
		delete(c.entries, key)
	}
	c.notify()
	resident, closeResident := c.resident, c.residentSet && !c.residentClose
	c.residentClose = true
	c.mu.Unlock()
	var result error
	for _, entry := range entries {
		<-entry.ready
		entry.mu.Lock()
		result = errors.Join(result, closeEntry(context.WithoutCancel(ctx), entry))
		entry.mu.Unlock()
	}
	if closeResident {
		result = errors.Join(result, closeModel(context.WithoutCancel(ctx), resident))
	}
	return result
}

func (c *ModelSessionDirector[Input, Model, Output]) oldestIdle() (sessionKey, *sessionEntry[Model]) {
	var oldestKey sessionKey
	var oldest *sessionEntry[Model]
	for key, entry := range c.entries {
		if entry.borrowers == 0 && (oldest == nil || entry.used < oldest.used) {
			oldestKey, oldest = key, entry
		}
	}
	return oldestKey, oldest
}

func (c *ModelSessionDirector[Input, Model, Output]) retire(ctx context.Context, key sessionKey, entry *sessionEntry[Model], cause error) error {
	c.mu.Lock()
	entry.dead = true
	entry.closing = true
	if c.entries[key] == entry {
		delete(c.entries, key)
		c.retirements++
	}
	c.notify()
	c.mu.Unlock()
	return errors.Join(cause, closeEntry(context.WithoutCancel(ctx), entry))
}

func (c *ModelSessionDirector[Input, Model, Output]) release(entry *sessionEntry[Model]) {
	entry.mu.Unlock()
	c.releaseBorrower(entry)
}

func (c *ModelSessionDirector[Input, Model, Output]) releaseBorrower(entry *sessionEntry[Model]) {
	c.mu.Lock()
	entry.borrowers--
	if entry.retiring {
		c.retiringUsers--
	}
	closeRetired := entry.borrowers == 0 && entry.dead && !entry.closing
	if closeRetired {
		entry.closing = true
	}
	c.notify()
	c.mu.Unlock()
	if closeRetired {
		c.finishRetirement(context.Background(), entry)
	}
}

func (c *ModelSessionDirector[Input, Model, Output]) supersedeLocked(
	scope artifact.ID,
	keep sessionKey,
) []*sessionEntry[Model] {
	if !scope.Valid() {
		return nil
	}
	var idle []*sessionEntry[Model]
	for key, entry := range c.entries {
		if key == keep || entry.scope != scope || entry.dead {
			continue
		}
		entry.dead = true
		entry.retiring = true
		delete(c.entries, key)
		c.retirements++
		c.retiring++
		c.retiringUsers += entry.borrowers
		select {
		case <-entry.ready:
			if entry.borrowers == 0 {
				entry.closing = true
				idle = append(idle, entry)
			}
		default:
		}
	}
	return idle
}

func (c *ModelSessionDirector[Input, Model, Output]) finishRetirement(ctx context.Context, entry *sessionEntry[Model]) error {
	entry.mu.Lock()
	err := closeEntry(ctx, entry)
	entry.mu.Unlock()
	c.mu.Lock()
	if entry.retiring {
		entry.retiring = false
		c.retiring--
	}
	c.notify()
	c.mu.Unlock()
	return err
}

func (c *ModelSessionDirector[Input, Model, Output]) notify() {
	close(c.changed)
	c.changed = make(chan struct{})
}

func closeEntry[Model any](ctx context.Context, entry *sessionEntry[Model]) error {
	if entry.closed {
		return nil
	}
	entry.closed = true
	return closeModel(ctx, entry.model)
}

func decodeJSONInput[Input any](
	name string,
	validate func(Input) error,
	modelID artifact.ID,
	program recipe.Program,
	raw string,
) (Input, artifact.Content, error) {
	var input Input
	if err := strictjson.DecodeBytes([]byte(raw), &input); err != nil {
		return input, artifact.Content{}, fmt.Errorf("decode %s input: %w", name, err)
	}
	if err := validate(input); err != nil {
		return input, artifact.Content{}, err
	}
	if program.Definition().Model != modelID {
		return input, artifact.Content{}, errors.New("capability runtime: program model differs from binding")
	}
	content, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindFile, "overgo."+name+"-input.v1"), input,
	)
	if err != nil {
		return input, artifact.Content{}, err
	}
	// A request over the inline bound embeds bulk conditioning -- raw
	// tensors, contexts, pixel buffers -- that is provenance the digest
	// carries, not knowledge the store archives (owner ruling
	// 2026-08-27). The durable document keeps the full request's hash
	// for identity and an abbreviated form for the reader; execution
	// identity derives from it, so replay semantics are unchanged for
	// any given request.
	if len(content.Data) > inlineRequestBytes {
		content, err = digestRequestContent(name, content)
	}
	return input, content, err
}

// inlineRequestBytes bounds the request document stored verbatim; a
// megabyte holds every settings-shaped request while excluding the
// embedded-tensor classes measured at tens of megabytes.
const inlineRequestBytes = 1 << 20

// digestRequestContent replaces an oversized request document with its
// durable footprint: the full request's digest and size for identity,
// and an abbreviated rendering for the reader.
func digestRequestContent(name string, full artifact.Content) (artifact.Content, error) {
	var document any
	if err := json.Unmarshal(full.Data, &document); err != nil {
		return artifact.Content{}, err
	}
	body := struct {
		RequestSHA256 string `json:"request_sha256"`
		RequestBytes  int    `json:"request_bytes"`
		Abbreviated   any    `json:"abbreviated"`
	}{
		RequestSHA256: full.Descriptor.ID.String(),
		RequestBytes:  len(full.Data),
		Abbreviated:   jsonabbrev.Value(document),
	}
	return artifact.JSONContent(
		artifact.JSONContract(artifact.KindFile, "overgo."+name+"-input-digest.v1"), body,
	)
}

func decodeScalarInput[Input any](
	name string,
	validate func(Input) error,
	modelID artifact.ID,
	program recipe.Program,
	raw string,
) (Input, artifact.Content, error) {
	input, content, err := decodeJSONInput(name, validate, modelID, program, raw)
	if err != nil {
		return input, content, err
	}
	if err := validateScalarProgram(modelID, program); err != nil {
		return input, artifact.Content{}, err
	}
	return input, content, nil
}

func executeScalar[Input, Model, Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	content artifact.Content,
	input Input,
	model Model,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) (Output, []workflowruntime.NodeWall, error) {
	definition := program.Definition()
	inputPort := definition.Inputs[0]
	return ExecuteMeasured[Output](
		ctx, store, modelID, program,
		"recipe/run/"+definition.ID.String()+"/"+content.Descriptor.ID.String(),
		map[recipe.PortName]workflowruntime.Value{
			inputPort.Name: workflowruntime.ArtifactValue(inputPort.Data, input, content),
		},
		func(runtime *workflowruntime.Runtime) error { return bind(runtime, modelID, model) },
	)
}

func validateScalarProgram(modelID artifact.ID, program recipe.Program) error {
	definition := program.Definition()
	if definition.Model != modelID || len(definition.Inputs) != 1 {
		return errors.New("capability runtime: scalar program identity differs")
	}
	return nil
}

func closeModel[Model any](ctx context.Context, model Model) error {
	if closer, ok := any(model).(interface{ Close(context.Context) error }); ok {
		return closer.Close(ctx)
	}
	return nil
}
