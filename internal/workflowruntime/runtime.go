package workflowruntime

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

const executionFailureCode = "execution_failed"

// Datum: runtime value plus optional provenance fact.
type Datum struct {
	Artifact artifact.Descriptor
	Content  *artifact.Content
	Value    any
}

// Value: typed port payload.
type Value struct {
	Kind  recipe.DataKind
	Items []Datum
}

// Single returns the only datum in a scalar value.
func (v Value) Single() (Datum, bool) {
	if len(v.Items) != 1 {
		return Datum{}, false
	}
	return v.Items[0], true
}

// ArtifactValue binds a runtime value to durable content.
func ArtifactValue(kind recipe.DataKind, value any, content artifact.Content) Value {
	return Value{Kind: kind, Items: []Datum{{
		Artifact: content.Descriptor, Content: &content, Value: value,
	}}}
}

// StepRequest: one compiled module invocation.
type StepRequest struct {
	Model  artifact.ID
	Inputs map[recipe.PortName]Value
}

// Adapter: module execution boundary.
type Adapter interface {
	Execute(context.Context, StepRequest) (map[recipe.PortName]Value, error)
}

type AdapterFunc func(context.Context, StepRequest) (map[recipe.PortName]Value, error)

func (f AdapterFunc) Execute(ctx context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
	return f(ctx, request)
}

// Result: workflow outputs plus durable run identity.
type Result struct {
	Outputs map[recipe.PortName]Value
	Run     runrecord.Run
	Commit  artifact.CommitID
}

// Runtime: registered module adapters plus run repository.
type Runtime struct {
	mu       sync.RWMutex
	store    artifact.Repository
	catalog  *recipe.Catalog
	adapters map[recipe.ModuleID]Adapter
}

func New(store artifact.Repository) (*Runtime, error) {
	return NewWithCatalog(store, workflowrecipe.Catalog())
}

// NewWithCatalog binds execution and adapter admission to one module catalog.
func NewWithCatalog(store artifact.Repository, catalog *recipe.Catalog) (*Runtime, error) {
	if store == nil {
		return nil, errors.New("workflow runtime: nil repository")
	}
	if catalog == nil {
		return nil, errors.New("workflow runtime: nil module catalog")
	}
	return &Runtime{
		store: store, catalog: catalog.Clone(), adapters: make(map[recipe.ModuleID]Adapter),
	}, nil
}

func (r *Runtime) Register(module recipe.ModuleID, adapter Adapter) error {
	if r == nil || r.catalog == nil || adapter == nil {
		return errors.New("workflow runtime: nil runtime or adapter")
	}
	if _, ok := r.catalog.Module(module); !ok {
		return fmt.Errorf("workflow runtime: unknown module %q", module)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, duplicate := r.adapters[module]; duplicate {
		return fmt.Errorf("workflow runtime: duplicate adapter %q", module)
	}
	r.adapters[module] = adapter
	return nil
}

func (r *Runtime) Execute(
	ctx context.Context,
	key string,
	definition recipe.Definition,
	inputs map[recipe.PortName]Value,
) (Result, error) {
	if r == nil || r.store == nil || r.catalog == nil {
		return Result{}, errors.New("workflow runtime: nil runtime")
	}
	if ctx == nil {
		return Result{}, errors.New("workflow runtime: nil context")
	}
	plan, err := workflowrecipe.Compile(definition)
	if err != nil {
		return Result{}, err
	}
	if plan.Support != workflowrecipe.ExecutionRuntime {
		return Result{}, errors.New("workflow runtime: orchestration-only plan")
	}
	return r.executeProgram(ctx, key, plan.Program, inputs)
}

// ExecuteProgram runs a catalog-resolved recipe and records its lineage.
func (r *Runtime) ExecuteProgram(
	ctx context.Context,
	key string,
	program recipe.Program,
	inputs map[recipe.PortName]Value,
) (Result, error) {
	if r == nil || r.store == nil || r.catalog == nil {
		return Result{}, errors.New("workflow runtime: nil runtime")
	}
	if ctx == nil {
		return Result{}, errors.New("workflow runtime: nil context")
	}
	definition := program.Definition()
	if definition.Task == recipe.TaskTraining {
		return Result{}, errors.New("workflow runtime: orchestration-only plan")
	}
	validated, err := recipe.CompileProgram(definition, r.catalog)
	if err != nil {
		return Result{}, err
	}
	if !slices.EqualFunc(program.Stages(), validated.Stages(), func(left, right recipe.Stage) bool {
		return left.Node == right.Node && left.Module.ID == right.Module.ID
	}) {
		return Result{}, errors.New("workflow runtime: compiled program differs from recipe")
	}
	return r.executeProgram(ctx, key, validated, inputs)
}

func (r *Runtime) executeProgram(
	ctx context.Context,
	key string,
	program recipe.Program,
	inputs map[recipe.PortName]Value,
) (Result, error) {
	definition := program.Definition()
	stages := program.Stages()
	inputIDs, inputFacts, err := externalFacts(inputs, false)
	if err != nil {
		return Result{}, err
	}
	outputs, executeErr := r.executePlan(ctx, definition, stages, inputs)
	outcome, failure := runrecord.OutcomeSucceeded, ""
	if executeErr != nil {
		outputs = nil
		if errors.Is(executeErr, context.Canceled) || errors.Is(executeErr, context.DeadlineExceeded) {
			outcome = runrecord.OutcomeCancelled
		} else {
			outcome, failure = runrecord.OutcomeFailed, executionFailureCode
		}
	}
	outputIDs, outputFacts, factErr := externalFacts(outputs, outcome == runrecord.OutcomeSucceeded)
	if factErr != nil {
		executeErr = factErr
		outcome, failure, outputs, outputIDs, outputFacts =
			runrecord.OutcomeFailed, executionFailureCode, nil, nil, artifactFacts{}
	}
	run, err := runrecord.NewRun(definition.ID, outcome, inputIDs, outputIDs, failure)
	if err != nil {
		return Result{}, errors.Join(executeErr, err)
	}
	batch, err := executionBatch(key, definition, run, inputFacts, outputFacts)
	if err != nil {
		return Result{Outputs: outputs, Run: run}, errors.Join(executeErr, err)
	}
	commitContext := ctx
	if ctx.Err() != nil {
		commitContext = context.WithoutCancel(ctx)
	}
	commit, commitErr := artifact.CommitBatch(commitContext, r.store, batch)
	result := Result{Outputs: outputs, Run: run, Commit: commit}
	if commitErr != nil {
		return result, errors.Join(executeErr, commitErr)
	}
	return result, executeErr
}

func (r *Runtime) executePlan(
	ctx context.Context,
	definition recipe.Definition,
	stages []recipe.Stage,
	external map[recipe.PortName]Value,
) (map[recipe.PortName]Value, error) {
	bound := make(map[recipe.Endpoint][]Value)
	if len(external) != len(definition.Inputs) {
		return nil, errors.New("workflow runtime: external input set differs")
	}
	for _, input := range definition.Inputs {
		value, ok := external[input.Name]
		if !ok || value.Kind != input.Data {
			return nil, fmt.Errorf("workflow runtime: invalid input %q", input.Name)
		}
		bound[input.Target] = append(bound[input.Target], cloneValue(value))
	}
	for _, stage := range stages {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		step, module := stage.Node, stage.Module
		stepInputs := make(map[recipe.PortName]Value, len(module.Inputs))
		for _, port := range module.Inputs {
			value, err := mergeValues(port.Data, bound[recipe.Endpoint{Node: step.ID, Port: port.Name}])
			if err != nil || !cardinalityValid(port.Cardinality, len(value.Items)) {
				if err == nil {
					err = fmt.Errorf("cardinality %s rejects %d items", port.Cardinality, len(value.Items))
				}
				return nil, fmt.Errorf("workflow runtime: input %s.%s: %w", step.ID, port.Name, err)
			}
			stepInputs[port.Name] = value
		}
		adapter, ok := r.adapter(step.Module)
		if !ok {
			return nil, fmt.Errorf("workflow runtime: module %q has no adapter", step.Module)
		}
		produced, err := adapter.Execute(ctx, StepRequest{
			Model: definition.Model, Inputs: stepInputs,
		})
		if err != nil {
			return nil, fmt.Errorf("workflow runtime: step %q: %w", step.ID, err)
		}
		validated, err := validateOutputs(module, produced)
		if err != nil {
			return nil, fmt.Errorf("workflow runtime: step %q: %w", step.ID, err)
		}
		for _, edge := range definition.Edges {
			if edge.From.Node == step.ID {
				if value, ok := validated[edge.From.Port]; ok {
					bound[edge.To] = append(bound[edge.To], cloneValue(value))
				}
			}
		}
		for name, value := range validated {
			bound[recipe.Endpoint{Node: step.ID, Port: name}] = []Value{value}
		}
	}
	outputs := make(map[recipe.PortName]Value, len(definition.Outputs))
	for _, output := range definition.Outputs {
		values := bound[output.Source]
		value, err := mergeValues(output.Data, values)
		if err != nil || len(value.Items) == 0 {
			if err == nil {
				err = errors.New("output is empty")
			}
			return nil, fmt.Errorf("workflow runtime: output %q: %w", output.Name, err)
		}
		outputs[output.Name] = value
	}
	return outputs, nil
}

func (r *Runtime) adapter(module recipe.ModuleID) (Adapter, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	adapter, ok := r.adapters[module]
	return adapter, ok
}

func validateOutputs(module recipe.Module, values map[recipe.PortName]Value) (map[recipe.PortName]Value, error) {
	ports := make(map[recipe.PortName]recipe.Port, len(module.Outputs))
	for _, port := range module.Outputs {
		ports[port.Name] = port
	}
	for name := range values {
		if _, ok := ports[name]; !ok {
			return nil, fmt.Errorf("unknown output %q", name)
		}
	}
	result := make(map[recipe.PortName]Value, len(values))
	for _, port := range module.Outputs {
		value, ok := values[port.Name]
		if !ok {
			value = Value{Kind: port.Data}
		}
		if value.Kind != port.Data || !cardinalityValid(port.Cardinality, len(value.Items)) {
			return nil, fmt.Errorf("invalid output %q", port.Name)
		}
		if ok {
			result[port.Name] = cloneValue(value)
		}
	}
	return result, nil
}

func mergeValues(kind recipe.DataKind, values []Value) (Value, error) {
	result := Value{Kind: kind}
	for _, value := range values {
		if value.Kind != kind {
			return Value{}, fmt.Errorf("cannot merge %s with %s", kind, value.Kind)
		}
		result.Items = append(result.Items, value.Items...)
	}
	return result, nil
}

func cardinalityValid(cardinality recipe.Cardinality, count int) bool {
	switch cardinality {
	case recipe.CardinalityOne:
		return count == 1
	case recipe.CardinalityOptional:
		return count <= 1
	case recipe.CardinalityMany:
		return true
	case recipe.CardinalityOneOrMany:
		return count > 0
	default:
		return false
	}
}

type artifactFacts struct {
	descriptors []artifact.Descriptor
	contents    []artifact.Content
}

func externalFacts(values map[recipe.PortName]Value, require bool) ([]artifact.ID, artifactFacts, error) {
	descriptors := make(map[artifact.ID]artifact.Descriptor)
	contents := make(map[artifact.ID]artifact.Content)
	for _, value := range values {
		for _, item := range value.Items {
			descriptor, content, ok, err := datumFact(item)
			if err != nil {
				return nil, artifactFacts{}, err
			}
			if !ok {
				if require {
					return nil, artifactFacts{}, errors.New("workflow runtime: final output lacks artifact identity")
				}
				continue
			}
			if current, exists := descriptors[descriptor.ID]; exists && current != descriptor {
				return nil, artifactFacts{}, errors.New("workflow runtime: artifact descriptor conflict")
			}
			descriptors[descriptor.ID] = descriptor
			if content != nil {
				if current, exists := contents[descriptor.ID]; exists &&
					(current.Descriptor != content.Descriptor || !bytes.Equal(current.Data, content.Data)) {
					return nil, artifactFacts{}, errors.New("workflow runtime: artifact content conflict")
				}
				contents[descriptor.ID] = content.Clone()
			}
		}
	}
	ids := make([]artifact.ID, 0, len(descriptors))
	for id := range descriptors {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, func(left, right artifact.ID) int {
		return cmp.Compare(left.String(), right.String())
	})
	facts := artifactFacts{
		descriptors: make([]artifact.Descriptor, 0, len(ids)),
		contents:    make([]artifact.Content, 0, len(contents)),
	}
	for _, id := range ids {
		facts.descriptors = append(facts.descriptors, descriptors[id])
		if content, ok := contents[id]; ok {
			facts.contents = append(facts.contents, content)
		}
	}
	return ids, facts, nil
}

func datumFact(datum Datum) (artifact.Descriptor, *artifact.Content, bool, error) {
	if datum.Content != nil {
		content := datum.Content.Clone()
		if err := content.Validate(); err != nil {
			return artifact.Descriptor{}, nil, false, err
		}
		if datum.Artifact.ID.Valid() && datum.Artifact != content.Descriptor {
			return artifact.Descriptor{}, nil, false, errors.New("workflow runtime: datum content descriptor differs")
		}
		return content.Descriptor, &content, true, nil
	}
	if !datum.Artifact.ID.Valid() {
		return artifact.Descriptor{}, nil, false, nil
	}
	if err := datum.Artifact.Validate(); err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	return datum.Artifact, nil, true, nil
}

func executionBatch(
	key string,
	definition recipe.Definition,
	run runrecord.Run,
	inputs, outputs artifactFacts,
) (artifact.Batch, error) {
	recipeContent, err := definition.ArtifactContent()
	if err != nil {
		return artifact.Batch{}, err
	}
	runBatch, err := run.Batch(key)
	if err != nil {
		return artifact.Batch{}, err
	}
	runBatch.Contents = append(runBatch.Contents, recipeContent)
	runBatch.Contents = append(runBatch.Contents, inputs.contents...)
	runBatch.Contents = append(runBatch.Contents, outputs.contents...)
	runBatch.Artifacts = append(runBatch.Artifacts, inputs.descriptors...)
	runBatch.Artifacts = append(runBatch.Artifacts, outputs.descriptors...)
	for _, dependency := range definition.Dependencies {
		runBatch.Lineage = append(runBatch.Lineage, artifact.Lineage{
			Child: definition.ID, Parent: dependency.Artifact, Relation: artifact.RelationDependsOn,
		})
	}
	return runBatch, nil
}

func cloneValue(value Value) Value {
	value.Items = slices.Clone(value.Items)
	for index := range value.Items {
		if value.Items[index].Content != nil {
			content := value.Items[index].Content.Clone()
			value.Items[index].Content = &content
		}
	}
	return value
}
