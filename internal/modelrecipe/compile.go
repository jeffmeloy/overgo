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
	return catalog.Clone()
}

func Inference(modelID artifact.ID, placement recipe.Placement) (recipe.Definition, error) {
	return inference(
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, placement,
	)
}

func InferenceWithProfile(
	modelID artifact.ID,
	profileID artifact.ID,
	placement recipe.Placement,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
	}, placement)
}

func InferenceWithModelDefinition(
	modelID artifact.ID,
	profileID artifact.ID,
	definitionID artifact.ID,
	placement recipe.Placement,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
		{Role: recipe.DependencyDefinition, Artifact: definitionID},
	}, placement)
}

func inference(dependencies []recipe.Dependency, placement recipe.Placement) (recipe.Definition, error) {
	compile := recipe.Node{ID: "compile", Module: ModuleCompileModelPlan, Placement: placement}
	forward := recipe.Node{ID: "forward", Module: ModuleForwardTokens, Placement: placement}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		dependencies,
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
	return definition.ArtifactContent()
}

func Batch(key string, definition recipe.Definition) (artifact.Batch, error) {
	content, err := Content(definition)
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, nil, nil)
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
