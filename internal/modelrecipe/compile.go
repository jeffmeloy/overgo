package modelrecipe

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
)

const (
	ModuleCompileModelPlan  recipe.ModuleID = "model.compile-plan"
	ModuleCompileDecodePlan recipe.ModuleID = "model.compile-decode-plan"
	ModuleForwardTokens     recipe.ModuleID = "model.forward-tokens"
)

var catalog = mustCatalog()

type Plan struct {
	Recipe recipe.Definition
	Model  model.ModelPlan
	Decode DecodePlan
	Nodes  []recipe.Node
}

// DecodeSessionPolicy: compiled decode-graph lifetime.
type DecodeSessionPolicy uint8

const (
	DecodeSessionRequest DecodeSessionPolicy = iota
	DecodeSessionCapacity
)

// DecodePlan: recipe-owned session program.
type DecodePlan struct {
	Session DecodeSessionPolicy
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
	decode := recipe.Node{ID: "decode", Module: ModuleCompileDecodePlan, Placement: placement}
	forward := recipe.Node{ID: "forward", Module: ModuleForwardTokens, Placement: placement}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		dependencies,
		[]recipe.Node{compile, decode, forward},
		[]recipe.Edge{
			{
				From: recipe.Endpoint{Node: compile.ID, Port: "plan"},
				To:   recipe.Endpoint{Node: decode.ID, Port: "plan"},
			},
			{
				From: recipe.Endpoint{Node: compile.ID, Port: "plan"},
				To:   recipe.Endpoint{Node: forward.ID, Port: "plan"},
			},
			{
				From: recipe.Endpoint{Node: decode.ID, Port: "session"},
				To:   recipe.Endpoint{Node: forward.ID, Port: "session"},
			},
		},
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

// CompileRuntime: canonical in-process inference program.
func CompileRuntime(spec model.Spec, weights model.Weights) (Plan, error) {
	modelPlan, err := model.CompileModelPlan(spec, weights)
	if err != nil {
		return Plan{}, err
	}
	nodes := runtimeNodes(recipe.PlacementHybrid)
	return compilePlan(recipe.Definition{}, nodes, modelPlan), nil
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
	foundCompile, foundDecode, foundForward := false, false, false
	for _, node := range definition.Nodes {
		foundCompile = foundCompile || node.Module == ModuleCompileModelPlan
		foundDecode = foundDecode || node.Module == ModuleCompileDecodePlan
		foundForward = foundForward || node.Module == ModuleForwardTokens
	}
	if !foundCompile || !foundDecode || !foundForward {
		return Plan{}, errors.New("model recipe: inference path is incomplete")
	}
	modelPlan, err := compileModel()
	if err != nil {
		return Plan{}, err
	}
	return compilePlan(definition, definition.Nodes, modelPlan), nil
}

func compilePlan(definition recipe.Definition, nodes []recipe.Node, modelPlan model.ModelPlan) Plan {
	session := DecodeSessionRequest
	if modelPlan.SupportsCapacityCache() {
		session = DecodeSessionCapacity
	}
	return Plan{
		Recipe: definition, Model: modelPlan, Decode: DecodePlan{Session: session},
		Nodes: append([]recipe.Node(nil), nodes...),
	}
}

func runtimeNodes(placement recipe.Placement) []recipe.Node {
	return []recipe.Node{
		{ID: "compile", Module: ModuleCompileModelPlan, Placement: placement},
		{ID: "decode", Module: ModuleCompileDecodePlan, Placement: placement},
		{ID: "forward", Module: ModuleForwardTokens, Placement: placement},
	}
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
			ID: ModuleCompileDecodePlan, Tasks: []recipe.Task{recipe.TaskInference}, Placements: placements,
			Inputs:  []recipe.Port{{Name: "plan", Data: recipe.DataModelPlan, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleForwardTokens, Tasks: []recipe.Task{recipe.TaskInference}, Placements: placements,
			Inputs: []recipe.Port{
				{Name: "plan", Data: recipe.DataModelPlan, Cardinality: recipe.CardinalityOne},
				{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne},
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
