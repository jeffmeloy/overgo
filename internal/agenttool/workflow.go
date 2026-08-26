package agenttool

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const toolWorkflowMaxSteps = 256

// ToolWorkflowStep is model-authored graph intent over one exact manual.
// After carries control dependencies only; tool values remain typed calls and
// results instead of becoming an untyped expression language.
type ToolWorkflowStep struct {
	ID     recipe.NodeID   `json:"id"`
	Manual artifact.ID     `json:"manual"`
	After  []recipe.NodeID `json:"after,omitempty"`
}

// ToolWorkflowProposal is an untrusted closed-world graph proposal.
type ToolWorkflowProposal struct {
	Model artifact.ID        `json:"model"`
	Steps []ToolWorkflowStep `json:"steps"`
}

// CompiledToolWorkflow binds the existing recipe program to its exact manual
// adapters. Its maps are private and only returned through cloning accessors.
type CompiledToolWorkflow struct {
	program recipe.Program
	manuals map[recipe.ModuleID]Manual
	steps   map[recipe.NodeID]recipe.ModuleID
}

// Program returns the validated immutable recipe program.
func (workflow CompiledToolWorkflow) Program() recipe.Program { return workflow.program }

// Manual returns the exact manual bound to one compiled module.
func (workflow CompiledToolWorkflow) Manual(module recipe.ModuleID) (Manual, bool) {
	manual, found := workflow.manuals[module]
	manual.Arguments = slices.Clone(manual.Arguments)
	manual.Transport.Args = slices.Clone(manual.Transport.Args)
	return manual, found
}

// Modules returns canonical module identities for adapter registration.
func (workflow CompiledToolWorkflow) Modules() []recipe.ModuleID {
	modules := slices.Collect(maps.Keys(workflow.manuals))
	slices.Sort(modules)
	return modules
}

// Calls validates one exact call per compiled step and returns recipe inputs.
func (workflow CompiledToolWorkflow) Calls(arguments map[recipe.NodeID]json.RawMessage) (map[recipe.PortName]recipe.ToolCall, error) {
	if len(arguments) != len(workflow.steps) {
		return nil, errors.New("agent tool: workflow arguments do not cover every step")
	}
	calls := make(map[recipe.PortName]recipe.ToolCall, len(arguments))
	for step, module := range workflow.steps {
		arguments, found := arguments[step]
		if !found {
			return nil, fmt.Errorf("agent tool: workflow step %q arguments are absent", step)
		}
		manual := workflow.manuals[module]
		if err := validateArguments(manual, arguments); err != nil {
			return nil, err
		}
		call, err := recipe.NewToolCall("call."+string(step), module, arguments)
		if err != nil {
			return nil, err
		}
		calls[toolWorkflowCallInput(step)] = call
	}
	return calls, nil
}

// CompileToolWorkflow validates a closed set of exact manuals and compiles
// dependency ordering through the existing recipe graph compiler.
func CompileToolWorkflow(proposal ToolWorkflowProposal, manuals []Manual) (CompiledToolWorkflow, error) {
	if proposal.Model.Kind() != artifact.KindModel || len(proposal.Steps) == 0 || len(proposal.Steps) > toolWorkflowMaxSteps {
		return CompiledToolWorkflow{}, errors.New("agent tool: invalid tool workflow boundary")
	}
	if err := validateManualSet(manuals); err != nil {
		return CompiledToolWorkflow{}, err
	}
	manualByID := make(map[artifact.ID]Manual, len(manuals))
	for _, manual := range manuals {
		manualByID[manual.ID] = manual
	}
	steps := slices.Clone(proposal.Steps)
	for index := range steps {
		steps[index].After = slices.Clone(steps[index].After)
		slices.Sort(steps[index].After)
		compact := slices.Compact(steps[index].After)
		if len(compact) != len(steps[index].After) {
			return CompiledToolWorkflow{}, errors.New("agent tool: duplicate workflow dependency")
		}
		steps[index].After = slices.Clone(compact)
	}
	sort.Slice(steps, func(left, right int) bool { return steps[left].ID < steps[right].ID })
	stepIndex := make(map[recipe.NodeID]ToolWorkflowStep, len(steps))
	manualUse := make(map[artifact.ID]bool, len(steps))
	for _, step := range steps {
		manual, found := manualByID[step.Manual]
		if step.ID == "" || !found {
			return CompiledToolWorkflow{}, errors.New("agent tool: workflow step or manual is absent")
		}
		if _, duplicate := stepIndex[step.ID]; duplicate || manualUse[manual.ID] {
			return CompiledToolWorkflow{}, errors.New("agent tool: workflow repeats a step or manual")
		}
		stepIndex[step.ID], manualUse[manual.ID] = step, true
	}
	if len(manualUse) != len(manualByID) {
		return CompiledToolWorkflow{}, errors.New("agent tool: workflow manual set is not closed")
	}
	for _, step := range steps {
		for _, dependency := range step.After {
			if dependency == step.ID || stepIndex[dependency].ID == "" {
				return CompiledToolWorkflow{}, fmt.Errorf("agent tool: workflow step %q has an invalid dependency", step.ID)
			}
		}
	}

	dependencies := []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: proposal.Model}}
	modules := make([]recipe.Module, 0, len(steps))
	nodes := make([]recipe.Node, 0, len(steps))
	edges := make([]recipe.Edge, 0, len(steps))
	inputs := make([]recipe.Input, 0, len(steps))
	outputs := make([]recipe.Output, 0, len(steps))
	boundManuals := make(map[recipe.ModuleID]Manual, len(steps))
	boundSteps := make(map[recipe.NodeID]recipe.ModuleID, len(steps))
	for slot, step := range steps {
		manual := manualByID[step.Manual]
		moduleID := recipe.ModuleID(manual.Name)
		moduleInputs := []recipe.Port{{Name: recipe.ToolCallPort, Data: recipe.DataToolCall, Cardinality: recipe.CardinalityOne}}
		for _, dependency := range step.After {
			moduleInputs = append(moduleInputs, recipe.Port{
				Name: toolWorkflowAfterPort(dependency), Data: recipe.DataToolResult, Cardinality: recipe.CardinalityOne,
			})
			edges = append(edges, recipe.Edge{
				From: recipe.Endpoint{Node: dependency, Port: recipe.ToolResultPort},
				To:   recipe.Endpoint{Node: step.ID, Port: toolWorkflowAfterPort(dependency)},
			})
		}
		modules = append(modules, recipe.Module{
			ID: moduleID, Tasks: []recipe.Task{recipe.TaskInference}, Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:  moduleInputs,
			Outputs: []recipe.Port{{Name: recipe.ToolResultPort, Data: recipe.DataToolResult, Cardinality: recipe.CardinalityOne}},
		})
		nodes = append(nodes, recipe.Node{ID: step.ID, Module: moduleID, Placement: recipe.PlacementHost})
		inputs = append(inputs, recipe.Input{
			Name: toolWorkflowCallInput(step.ID), Data: recipe.DataToolCall,
			Target: recipe.Endpoint{Node: step.ID, Port: recipe.ToolCallPort},
		})
		outputs = append(outputs, recipe.Output{
			Name: toolWorkflowResultOutput(step.ID), Data: recipe.DataToolResult,
			Source: recipe.Endpoint{Node: step.ID, Port: recipe.ToolResultPort},
		})
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyToolManual, Slot: uint32(slot), Artifact: manual.ID,
		})
		boundManuals[moduleID], boundSteps[step.ID] = manual, moduleID
	}
	catalog, err := recipe.NewCatalog(modules...)
	if err != nil {
		return CompiledToolWorkflow{}, err
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference, dependencies, nodes, edges, inputs, outputs,
	)
	if err != nil {
		return CompiledToolWorkflow{}, err
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		return CompiledToolWorkflow{}, err
	}
	if err := validateMutationInspectionOrder(steps, stepIndex, manualByID); err != nil {
		return CompiledToolWorkflow{}, err
	}
	return CompiledToolWorkflow{program: program, manuals: boundManuals, steps: boundSteps}, nil
}

func validateMutationInspectionOrder(
	steps []ToolWorkflowStep,
	stepIndex map[recipe.NodeID]ToolWorkflowStep,
	manuals map[artifact.ID]Manual,
) error {
	var hasInspectionAncestor func(recipe.NodeID, map[recipe.NodeID]bool) bool
	hasInspectionAncestor = func(id recipe.NodeID, visited map[recipe.NodeID]bool) bool {
		if visited[id] {
			return false
		}
		visited[id] = true
		for _, prior := range stepIndex[id].After {
			step := stepIndex[prior]
			if manuals[step.Manual].Effect == EffectInspection || hasInspectionAncestor(prior, visited) {
				return true
			}
		}
		return false
	}
	for _, step := range steps {
		if manuals[step.Manual].Effect == EffectMutation && !hasInspectionAncestor(step.ID, map[recipe.NodeID]bool{}) {
			return fmt.Errorf("agent tool: mutation step %q has no inspection ancestor", step.ID)
		}
	}
	return nil
}

func toolWorkflowCallInput(step recipe.NodeID) recipe.PortName {
	return recipe.PortName("call." + string(step))
}

func toolWorkflowResultOutput(step recipe.NodeID) recipe.PortName {
	return recipe.PortName("result." + string(step))
}

func toolWorkflowAfterPort(step recipe.NodeID) recipe.PortName {
	return recipe.PortName("after." + string(step))
}
