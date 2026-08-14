package capabilityruntime

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

type sessionKey struct {
	model, recipe  artifact.ID
	device, policy string
}

type sessionEntry[Model any] struct {
	mu        sync.Mutex
	model     Model
	ready     chan struct{}
	loadErr   error
	used      uint64
	borrowers int
	dead      bool
	closed    bool
}

// ScalarSessionCache owns bounded reusable model sessions.
type ScalarSessionCache[Input, Model, Output any] struct {
	name     string
	device   string
	capacity int
	validate func(Input) error
	policy   func(Input) (string, error)
	load     func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error)
	reset    func(context.Context, Model, Input) error
	bind     func(*workflowruntime.Runtime, artifact.ID, Model) error

	mu      sync.Mutex
	tick    uint64
	entries map[sessionKey]*sessionEntry[Model]
	changed chan struct{}
	closed  bool
}

func NewScalarSessionCache[Input, Model, Output any](
	name, device string,
	capacity int,
	validate func(Input) error,
	policy func(Input) (string, error),
	load func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error),
	reset func(context.Context, Model, Input) error,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) (*ScalarSessionCache[Input, Model, Output], error) {
	if name == "" || device == "" || capacity <= 0 || validate == nil || policy == nil || load == nil || reset == nil || bind == nil {
		return nil, errors.New("capability runtime: incomplete scalar session cache")
	}
	return &ScalarSessionCache[Input, Model, Output]{
		name: name, device: device, capacity: capacity,
		validate: validate, policy: policy, load: load, reset: reset, bind: bind,
		entries: make(map[sessionKey]*sessionEntry[Model]), changed: make(chan struct{}),
	}, nil
}

func (c *ScalarSessionCache[Input, Model, Output]) Executor() Executor {
	return func(
		ctx context.Context,
		store artifact.Repository,
		path string,
		modelID artifact.ID,
		program recipe.Program,
		raw string,
	) (any, error) {
		return c.execute(ctx, store, path, modelID, program, raw)
	}
}

func (c *ScalarSessionCache[Input, Model, Output]) execute(
	ctx context.Context,
	store artifact.Repository,
	path string,
	modelID artifact.ID,
	program recipe.Program,
	raw string,
) (Output, error) {
	var zero Output
	input, content, err := decodeInput(c.name, c.validate, raw)
	if err != nil {
		return zero, err
	}
	if err := validateScalarProgram(modelID, program); err != nil {
		return zero, err
	}
	policy, err := c.policy(input)
	if err != nil || policy == "" {
		return zero, errors.Join(errors.New("capability runtime: invalid execution policy"), err)
	}
	definition := program.Definition()
	key := sessionKey{
		model: modelID, recipe: definition.ID, device: c.device, policy: policy,
	}

	entry, fresh, err := c.lease(ctx, store, path, key, program, input)
	if err != nil {
		return zero, err
	}
	defer c.release(entry)
	if !fresh {
		if err := c.reset(ctx, entry.model, input); err != nil {
			return zero, c.retire(ctx, key, entry, err)
		}
	}
	output, err := executeScalar[Input, Model, Output](
		ctx, store, modelID, program, content, input, entry.model, c.bind,
	)
	if err != nil {
		err = c.retire(ctx, key, entry, err)
	}
	return output, err
}

func (c *ScalarSessionCache[Input, Model, Output]) lease(
	ctx context.Context,
	store artifact.Repository,
	path string,
	key sessionKey,
	program recipe.Program,
	input Input,
) (*sessionEntry[Model], bool, error) {
	for {
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return nil, false, errors.New("capability runtime: session cache is closed")
		}
		c.tick++
		if entry, ok := c.entries[key]; ok {
			entry.borrowers++
			entry.used = c.tick
			c.mu.Unlock()
			select {
			case <-entry.ready:
			case <-ctx.Done():
				c.releaseBorrower(entry)
				return nil, false, ctx.Err()
			}
			if entry.loadErr != nil {
				c.releaseBorrower(entry)
				return nil, false, entry.loadErr
			}
			entry.mu.Lock()
			c.mu.Lock()
			dead := entry.dead
			c.mu.Unlock()
			if dead {
				entry.mu.Unlock()
				c.releaseBorrower(entry)
				continue
			}
			return entry, false, nil
		}
		if len(c.entries) >= c.capacity {
			oldestKey, oldest := c.oldestIdle()
			if oldest == nil {
				changed := c.changed
				c.mu.Unlock()
				select {
				case <-changed:
					continue
				case <-ctx.Done():
					return nil, false, ctx.Err()
				}
			}
			oldest.dead = true
			delete(c.entries, oldestKey)
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
		entry := &sessionEntry[Model]{ready: make(chan struct{}), used: c.tick, borrowers: 1}
		c.entries[key] = entry
		c.notify()
		c.mu.Unlock()
		model, err := c.load(ctx, store, path, program, input)
		entry.model, entry.loadErr = model, err
		close(entry.ready)
		if err != nil {
			c.mu.Lock()
			entry.dead = true
			if c.entries[key] == entry {
				delete(c.entries, key)
			}
			c.notify()
			c.mu.Unlock()
			c.releaseBorrower(entry)
			return nil, false, err
		}
		entry.mu.Lock()
		c.mu.Lock()
		dead := entry.dead
		c.mu.Unlock()
		if dead {
			entry.mu.Unlock()
			c.releaseBorrower(entry)
			return nil, false, errors.New("capability runtime: session cache is closed")
		}
		return entry, true, nil
	}
}

func (c *ScalarSessionCache[Input, Model, Output]) Close(ctx context.Context) error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	entries := make([]*sessionEntry[Model], 0, len(c.entries))
	for key, entry := range c.entries {
		entry.dead = true
		entries = append(entries, entry)
		delete(c.entries, key)
	}
	c.notify()
	c.mu.Unlock()
	var result error
	for _, entry := range entries {
		<-entry.ready
		entry.mu.Lock()
		result = errors.Join(result, closeEntry(context.WithoutCancel(ctx), entry))
		entry.mu.Unlock()
	}
	return result
}

func (c *ScalarSessionCache[Input, Model, Output]) oldestIdle() (sessionKey, *sessionEntry[Model]) {
	var oldestKey sessionKey
	var oldest *sessionEntry[Model]
	for key, entry := range c.entries {
		if entry.borrowers == 0 && (oldest == nil || entry.used < oldest.used) {
			oldestKey, oldest = key, entry
		}
	}
	return oldestKey, oldest
}

func (c *ScalarSessionCache[Input, Model, Output]) retire(ctx context.Context, key sessionKey, entry *sessionEntry[Model], cause error) error {
	c.mu.Lock()
	entry.dead = true
	if c.entries[key] == entry {
		delete(c.entries, key)
	}
	c.notify()
	c.mu.Unlock()
	return errors.Join(cause, closeEntry(context.WithoutCancel(ctx), entry))
}

func (c *ScalarSessionCache[Input, Model, Output]) release(entry *sessionEntry[Model]) {
	entry.mu.Unlock()
	c.releaseBorrower(entry)
}

func (c *ScalarSessionCache[Input, Model, Output]) releaseBorrower(entry *sessionEntry[Model]) {
	c.mu.Lock()
	entry.borrowers--
	c.notify()
	c.mu.Unlock()
}

func (c *ScalarSessionCache[Input, Model, Output]) notify() {
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

func decodeInput[Input any](name string, validate func(Input) error, raw string) (Input, artifact.Content, error) {
	var input Input
	if err := strictjson.DecodeBytes([]byte(raw), &input); err != nil {
		return input, artifact.Content{}, fmt.Errorf("decode %s input: %w", name, err)
	}
	if err := validate(input); err != nil {
		return input, artifact.Content{}, err
	}
	content, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindFile, "overgo."+name+"-input.v1"), input,
	)
	return input, content, err
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
) (Output, error) {
	definition := program.Definition()
	inputPort := definition.Inputs[0]
	return Execute[Output](
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
