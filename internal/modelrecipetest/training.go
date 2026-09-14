package modelrecipetest

import (
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// DPODefinition builds the shared preference-training fixture.
func DPODefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchPreference, Placement: recipe.PlacementHost},
		{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
		{ID: "reference", Module: workflowrecipe.ModuleScoreReference, Placement: recipe.PlacementHost, ModelSlot: 1},
		{ID: "objective", Module: workflowrecipe.ModuleDPOObjective, Placement: recipe.PlacementHost},
		{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
		{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("batch", "batch", "policy", "batch"), edge("batch", "batch", "reference", "batch"),
			edge("policy", "scores", "objective", "policy"), edge("reference", "scores", "objective", "reference"),
			edge("objective", "loss", "backward", "loss"), edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
}

// PolicyDependencies binds the fixture's declared training policies.
func PolicyDependencies(spec trainingprogram.PolicySpec) []recipe.Dependency {
	return []recipe.Dependency{
		{Role: recipe.DependencyObjective, Artifact: spec.Objective},
		{Role: recipe.DependencyPrecision, Artifact: spec.Precision},
		{Role: recipe.DependencyPlacement, Artifact: spec.Placement},
		{Role: recipe.DependencyMemory, Artifact: spec.Memory},
		{Role: recipe.DependencyOptimizer, Artifact: spec.Optimizer},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: spec.Checkpoint},
		{Role: recipe.DependencyEvaluation, Artifact: spec.Evaluation},
		{Role: recipe.DependencyPromotion, Artifact: spec.Promotion},
	}
}
