package modelrecipe

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
)

const (
	ModuleCompileModelPlan recipe.ModuleID = "model.compile-plan"
	ModuleForwardTokens    recipe.ModuleID = "model.forward-tokens"
)

var catalog = mustCatalog()

type Plan struct {
	Recipe recipe.Definition
	Model  model.ModelPlan
	Nodes  []recipe.Node
}

func Catalog() *recipe.Catalog {
	copy, err := recipe.NewCatalog(catalog.Modules()...)
	if err != nil {
		panic(err)
	}
	return copy
}

func Inference(modelID artifact.ID, placement recipe.Placement) (recipe.Definition, error) {
	compile := recipe.Node{ID: "compile", Module: ModuleCompileModelPlan, Placement: placement}
	forward := recipe.Node{ID: "forward", Module: ModuleForwardTokens, Placement: placement}
	return recipe.NewDefinition(
		recipe.TaskInference,
		modelID,
		[]recipe.Node{compile, forward},
		[]recipe.Edge{{
			From: recipe.Endpoint{Node: compile.ID, Port: "plan"},
			To:   recipe.Endpoint{Node: forward.ID, Port: "plan"},
		}},
		[]recipe.Input{{
			Name: "tokens", Data: recipe.DataTokens,
			Target: recipe.Endpoint{Node: forward.ID, Port: "tokens"},
		}},
		[]recipe.Output{{
			Name: "logits", Data: recipe.DataLogits,
			Source: recipe.Endpoint{Node: forward.ID, Port: "logits"},
		}},
	)
}

func Compile(definition recipe.Definition, spec model.Spec, weights model.Weights) (Plan, error) {
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlan(spec, weights)
	})
}

func compileDefinition(
	definition recipe.Definition,
	compileModel func() (model.ModelPlan, error),
) (Plan, error) {
	if err := definition.Validate(catalog); err != nil {
		return Plan{}, err
	}
	if definition.Task != recipe.TaskInference {
		return Plan{}, fmt.Errorf("model recipe: unsupported task %q", definition.Task)
	}
	foundCompile, foundForward := false, false
	for _, node := range definition.Nodes {
		foundCompile = foundCompile || node.Module == ModuleCompileModelPlan
		foundForward = foundForward || node.Module == ModuleForwardTokens
	}
	if !foundCompile || !foundForward {
		return Plan{}, errors.New("model recipe: inference path is incomplete")
	}
	modelPlan, err := compileModel()
	if err != nil {
		return Plan{}, err
	}
	return Plan{Recipe: definition, Model: modelPlan, Nodes: append([]recipe.Node(nil), definition.Nodes...)}, nil
}

func Content(definition recipe.Definition) (artifact.Content, error) {
	descriptor, err := definition.Descriptor()
	if err != nil {
		return artifact.Content{}, err
	}
	data, err := definition.Content()
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{Descriptor: descriptor, Data: data}, nil
}

func Batch(key string, definition recipe.Definition) (artifact.Batch, error) {
	content, err := Content(definition)
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.Batch{Key: key, Contents: []artifact.Content{content}}, nil
}

func mustCatalog() *recipe.Catalog {
	placements := []recipe.Placement{recipe.PlacementHost, recipe.PlacementDevice, recipe.PlacementHybrid}
	catalog, err := recipe.NewCatalog(
		recipe.Module{
			ID: ModuleCompileModelPlan, Tasks: []recipe.Task{recipe.TaskInference}, Placements: placements,
			Outputs: []recipe.Port{{Name: "plan", Data: recipe.DataModelPlan, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleForwardTokens, Tasks: []recipe.Task{recipe.TaskInference}, Placements: placements,
			Inputs: []recipe.Port{
				{Name: "plan", Data: recipe.DataModelPlan, Cardinality: recipe.CardinalityOne},
				{Name: "tokens", Data: recipe.DataTokens, Cardinality: recipe.CardinalityOne},
			},
			Outputs: []recipe.Port{{Name: "logits", Data: recipe.DataLogits, Cardinality: recipe.CardinalityOne}},
		},
	)
	if err != nil {
		panic(err)
	}
	return catalog
}
