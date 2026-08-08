package modelrecipe

import (
	"errors"
	"fmt"
	"slices"

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
	Identity ProgramIdentity
	Recipe   recipe.Definition
	Model    model.ModelPlan
	Decode   DecodePlan
	Nodes    []recipe.Node
}

// Runtime: compiled execution surface.
type Runtime string

const RuntimeInference Runtime = "inference"

// ProgramIdentity: exact serving bindings.
type ProgramIdentity struct {
	Model         artifact.ID
	Profile       artifact.ID
	Definition    artifact.ID
	Recipe        artifact.ID
	RecipeVersion uint16
	Placement     recipe.Placement
	Runtime       Runtime
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
		Identity: programIdentity(definition), Recipe: definition,
		Model: modelPlan, Decode: DecodePlan{Session: session},
		Nodes: append([]recipe.Node(nil), nodes...),
	}
}

func programIdentity(definition recipe.Definition) ProgramIdentity {
	identity := ProgramIdentity{
		Model: definition.Model, Recipe: definition.ID, RecipeVersion: definition.Version,
		Runtime: RuntimeInference,
	}
	identity.Profile, _ = definition.Dependency(recipe.DependencyProfile, 0)
	identity.Definition, _ = definition.Dependency(recipe.DependencyDefinition, 0)
	for _, node := range definition.Nodes {
		if node.Module == ModuleForwardTokens {
			identity.Placement = node.Placement
			break
		}
	}
	return identity
}

func validateProgramIdentity(definition recipe.Definition, identity ProgramIdentity) error {
	if err := definition.Validate(catalog); err != nil {
		return err
	}
	want := programIdentity(definition)
	if identity != want || identity.Model.Kind() != artifact.KindModel ||
		identity.Profile.Kind() != artifact.KindProfile ||
		identity.Definition.Kind() != artifact.KindModelDefinition ||
		identity.Recipe.Kind() != artifact.KindRecipe ||
		identity.Runtime != RuntimeInference || identity.Placement == "" {
		return errors.New("model recipe: serving program identity is incomplete")
	}
	for _, node := range definition.Nodes {
		if node.Placement != identity.Placement {
			return errors.New("model recipe: serving program has mixed placement")
		}
	}
	return nil
}

// ValidateServing: complete identity-bound inference program.
func (p Plan) ValidateServing() error {
	if err := validateProgramIdentity(p.Recipe, p.Identity); err != nil {
		return err
	}
	if p.Model.Profile().Name == "" || p.Model.LayerCount() == 0 || !p.Model.HasCacheSchemas() {
		return errors.New("model recipe: serving model program is incomplete")
	}
	if !slices.Equal(p.Nodes, p.Recipe.Nodes) {
		return errors.New("model recipe: serving node program differs")
	}
	return nil
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
