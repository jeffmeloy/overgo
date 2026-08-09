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
	// ModuleForecastSeries: host forward for series-forecast capability
	// packages; input series tensor, output quantile-forecast tensor.
	ModuleForecastSeries recipe.ModuleID = "model.forecast-series"
	// ModuleTabularPredict: host forward for tabular ICL capability
	// packages; input table tensor, output per-row predictions tensor.
	ModuleTabularPredict recipe.ModuleID = "model.tabular-predict"
	// ModuleSeq2SeqGenerate: host encoder-decoder generation for seq2seq
	// capability packages; input source tokens, output generated tokens.
	ModuleSeq2SeqGenerate recipe.ModuleID = "model.seq2seq-generate"
	// ModuleSpeechSynthesize: host text-to-speech for speech capability
	// packages; input text, output synthesized audio.
	ModuleSpeechSynthesize recipe.ModuleID = "model.speech-synthesize"
	// ModuleImageGenerate: host class-conditional image sampling for
	// image-generation capability packages; input condition tensor, output
	// generated image.
	ModuleImageGenerate recipe.ModuleID = "model.image-generate"
	// ModuleVideoGenerate: text-to-video generation (device denoise session
	// + CUDA VAE decode); prompt text in, decoded video frames out.
	ModuleVideoGenerate recipe.ModuleID = "model.video-generate"
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
type DecodeSessionPolicy = recipe.SessionPolicy

const (
	DecodeSessionRequest  = recipe.SessionRequest
	DecodeSessionCapacity = recipe.SessionCapacity
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
	session DecodeSessionPolicy,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
		{Role: recipe.DependencyDefinition, Artifact: definitionID},
	}, placement, session)
}

// ForecastDefinition: single host forecast node bound to the model artifact.
// The capability package derives every dimension from the artifact itself,
// so the definition carries no profile document.
func ForecastDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "forecast", Module: ModuleForecastSeries, Placement: recipe.PlacementHost}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskForecast,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "series", Data: recipe.DataTensor,
			Target: recipe.Endpoint{Node: forward.ID, Port: "series"},
		}},
		[]recipe.Output{{
			Name: "forecast", Data: recipe.DataTensor,
			Source: recipe.Endpoint{Node: forward.ID, Port: "forecast"},
		}},
	)
}

// TabularDefinition: single host tabular-predict node bound to the model
// artifact. The capability package derives every dimension from the
// artifact's tensor shapes, so the definition carries no profile document.
func TabularDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "tabular", Module: ModuleTabularPredict, Placement: recipe.PlacementHost}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTabular,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "table", Data: recipe.DataTensor,
			Target: recipe.Endpoint{Node: forward.ID, Port: "table"},
		}},
		[]recipe.Output{{
			Name: "predictions", Data: recipe.DataTensor,
			Source: recipe.Endpoint{Node: forward.ID, Port: "predictions"},
		}},
	)
}

// Seq2SeqDefinition: single host seq2seq-generate node bound to the model
// artifact. The capability package derives every dimension from the
// artifact's tensor shapes, so the definition carries no profile document.
func Seq2SeqDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "seq2seq", Module: ModuleSeq2SeqGenerate, Placement: recipe.PlacementHost}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskSeq2Seq,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "source", Data: recipe.DataTokens,
			Target: recipe.Endpoint{Node: forward.ID, Port: "source"},
		}},
		[]recipe.Output{{
			Name: "tokens", Data: recipe.DataTokens,
			Source: recipe.Endpoint{Node: forward.ID, Port: "tokens"},
		}},
	)
}

// SpeechDefinition: single host speech-synthesize node bound to the model
// artifact. The capability package derives every dimension from the
// artifact's tensor shapes, so the definition carries no profile document.
func SpeechDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "speech", Module: ModuleSpeechSynthesize, Placement: recipe.PlacementHost}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskSpeech,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "text", Data: recipe.DataText,
			Target: recipe.Endpoint{Node: forward.ID, Port: "text"},
		}},
		[]recipe.Output{{
			Name: "audio", Data: recipe.DataAudio,
			Source: recipe.Endpoint{Node: forward.ID, Port: "audio"},
		}},
	)
}

// ImageGenDefinition: single host image-generate node bound to the model
// artifact. The capability package derives every dimension from the
// artifact's tensor lengths, so the definition carries no profile document.
func ImageGenDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "imagegen", Module: ModuleImageGenerate, Placement: recipe.PlacementHost}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskImageGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "condition", Data: recipe.DataTensor,
			Target: recipe.Endpoint{Node: forward.ID, Port: "condition"},
		}},
		[]recipe.Output{{
			Name: "image", Data: recipe.DataImage,
			Source: recipe.Endpoint{Node: forward.ID, Port: "image"},
		}},
	)
}

// VideoGenDefinition: single video-generate node bound to the model
// artifact (device placement — the production denoise/decode path is the
// CUDA session). The capability package derives dimensions from the
// artifact, so the definition carries no profile document.
func VideoGenDefinition(modelID artifact.ID) (recipe.Definition, error) {
	forward := recipe.Node{ID: "videogen", Module: ModuleVideoGenerate, Placement: recipe.PlacementDevice}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskVideoGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{forward},
		nil,
		[]recipe.Input{{
			Name: "prompt", Data: recipe.DataText,
			Target: recipe.Endpoint{Node: forward.ID, Port: "prompt"},
		}},
		[]recipe.Output{{
			Name: "video", Data: recipe.DataVideo,
			Source: recipe.Endpoint{Node: forward.ID, Port: "video"},
		}},
	)
}

func inference(
	dependencies []recipe.Dependency,
	placement recipe.Placement,
	session DecodeSessionPolicy,
) (recipe.Definition, error) {
	compile := recipe.Node{ID: "compile", Module: ModuleCompileModelPlan, Placement: placement}
	decode := recipe.Node{
		ID: "decode", Module: ModuleCompileDecodePlan, Placement: placement, Session: session,
	}
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
	decode, err := compileDecodePlan(definition.Nodes, modelPlan)
	if err != nil {
		return Plan{}, err
	}
	return compilePlan(definition, definition.Nodes, modelPlan, decode), nil
}

func compilePlan(
	definition recipe.Definition,
	nodes []recipe.Node,
	modelPlan model.ModelPlan,
	decode DecodePlan,
) Plan {
	return Plan{
		Identity: programIdentity(definition), Recipe: definition,
		Model: modelPlan, Decode: decode,
		Nodes: append([]recipe.Node(nil), nodes...),
	}
}

func compileDecodePlan(nodes []recipe.Node, modelPlan model.ModelPlan) (DecodePlan, error) {
	var session DecodeSessionPolicy
	for _, node := range nodes {
		if node.Module == ModuleCompileDecodePlan {
			if session != "" {
				return DecodePlan{}, errors.New("model recipe: multiple decode-session policies")
			}
			session = node.Session
			continue
		}
		if node.Session != "" {
			return DecodePlan{}, fmt.Errorf(
				"model recipe: module %q carries decode-session policy", node.Module,
			)
		}
	}
	if session == "" {
		return DecodePlan{}, errors.New("model recipe: decode-session policy is missing")
	}
	if session == DecodeSessionCapacity && !modelPlan.SupportsCapacityCache() {
		return DecodePlan{}, errors.New("model recipe: capacity session is incompatible with model plan")
	}
	return DecodePlan{Session: session}, nil
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
	decode, err := compileDecodePlan(p.Nodes, p.Model)
	if err != nil || decode != p.Decode {
		return errors.New("model recipe: serving decode program differs")
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
		recipe.Module{
			ID: ModuleForecastSeries, Tasks: []recipe.Task{recipe.TaskForecast},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:     []recipe.Port{{Name: "series", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "forecast", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleTabularPredict, Tasks: []recipe.Task{recipe.TaskTabular},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:     []recipe.Port{{Name: "table", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "predictions", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleSeq2SeqGenerate, Tasks: []recipe.Task{recipe.TaskSeq2Seq},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:     []recipe.Port{{Name: "source", Data: recipe.DataTokens, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "tokens", Data: recipe.DataTokens, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleSpeechSynthesize, Tasks: []recipe.Task{recipe.TaskSpeech},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:     []recipe.Port{{Name: "text", Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleImageGenerate, Tasks: []recipe.Task{recipe.TaskImageGen},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs:     []recipe.Port{{Name: "condition", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "image", Data: recipe.DataImage, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleVideoGenerate, Tasks: []recipe.Task{recipe.TaskVideoGen},
			Placements: []recipe.Placement{recipe.PlacementDevice},
			Inputs:     []recipe.Port{{Name: "prompt", Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "video", Data: recipe.DataVideo, Cardinality: recipe.CardinalityOne}},
		},
	)
	if err != nil {
		panic(err)
	}
	return catalog
}
