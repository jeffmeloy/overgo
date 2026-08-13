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
	model, recipe, device, policy string
}

type sessionEntry[Model any] struct {
	model Model
	used  uint64
}

// ScalarSessionCache owns bounded reusable model sessions.
type ScalarSessionCache[Input, Model, Output any] struct {
	name     string
	device   string
	capacity int
	validate func(Input) error
	policy   func(Input) (string, error)
	load     func(context.Context, string, Input) (Model, error)
	reset    func(context.Context, Model, Input) error
	bind     func(*workflowruntime.Runtime, artifact.ID, Model) error

	mu      sync.Mutex
	tick    uint64
	entries map[sessionKey]sessionEntry[Model]
}

func NewScalarSessionCache[Input, Model, Output any](
	name, device string,
	capacity int,
	validate func(Input) error,
	policy func(Input) (string, error),
	load func(context.Context, string, Input) (Model, error),
	reset func(context.Context, Model, Input) error,
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) (*ScalarSessionCache[Input, Model, Output], error) {
	if name == "" || device == "" || capacity <= 0 || validate == nil || policy == nil || load == nil || reset == nil || bind == nil {
		return nil, errors.New("capability runtime: incomplete scalar session cache")
	}
	return &ScalarSessionCache[Input, Model, Output]{
		name: name, device: device, capacity: capacity,
		validate: validate, policy: policy, load: load, reset: reset, bind: bind,
		entries: make(map[sessionKey]sessionEntry[Model]),
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
	policy, err := c.policy(input)
	if err != nil || policy == "" {
		return zero, errors.Join(errors.New("capability runtime: invalid execution policy"), err)
	}
	definition := program.Definition()
	key := sessionKey{
		model: modelID.String(), recipe: definition.ID.String(), device: c.device, policy: policy,
	}

	// A cached model is a mutable session. Serialize lease, execution, reset.
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tick++
	entry, hit := c.entries[key]
	if hit {
		if err := c.reset(ctx, entry.model, input); err != nil {
			_ = closeModel(context.WithoutCancel(ctx), entry.model)
			delete(c.entries, key)
			return zero, err
		}
	} else {
		if err := c.evict(ctx); err != nil {
			return zero, err
		}
		model, err := c.load(ctx, path, input)
		if err != nil {
			return zero, err
		}
		entry = sessionEntry[Model]{model: model}
	}
	entry.used = c.tick
	c.entries[key] = entry
	output, err := executeScalar[Input, Model, Output](
		ctx, store, modelID, program, content, input, entry.model, c.bind,
	)
	if err != nil {
		_ = closeModel(context.WithoutCancel(ctx), entry.model)
		delete(c.entries, key)
	}
	return output, err
}

func (c *ScalarSessionCache[Input, Model, Output]) evict(ctx context.Context) error {
	if len(c.entries) < c.capacity {
		return nil
	}
	var oldest sessionKey
	oldestTick := ^uint64(0)
	for key, entry := range c.entries {
		if entry.used < oldestTick {
			oldest, oldestTick = key, entry.used
		}
	}
	entry := c.entries[oldest]
	delete(c.entries, oldest)
	return closeModel(context.WithoutCancel(ctx), entry.model)
}

func (c *ScalarSessionCache[Input, Model, Output]) Close(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	var result error
	for key, entry := range c.entries {
		result = errors.Join(result, closeModel(ctx, entry.model))
		delete(c.entries, key)
	}
	return result
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
	var zero Output
	definition := program.Definition()
	if definition.Model != modelID || len(definition.Inputs) != 1 {
		return zero, errors.New("capability runtime: scalar program identity differs")
	}
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

func closeModel[Model any](ctx context.Context, model Model) error {
	if closer, ok := any(model).(interface{ Close(context.Context) error }); ok {
		return closer.Close(ctx)
	}
	return nil
}
