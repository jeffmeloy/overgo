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
	ModuleSeq2SeqEncode  recipe.ModuleID = "model.seq2seq-encode"
	ModuleSeq2SeqPrepare recipe.ModuleID = "model.seq2seq-prepare"
	ModuleSeq2SeqSelect  recipe.ModuleID = "model.seq2seq-select"
	ModuleSpeechTokenize recipe.ModuleID = "model.speech-tokenize"
	ModuleSpeechGenerate recipe.ModuleID = "model.speech-generate"
	ModuleSpeechDecode   recipe.ModuleID = "model.speech-decode"
	ModuleImagePrepare   recipe.ModuleID = "model.image-prepare"
	ModuleImageIntegrate recipe.ModuleID = "model.image-integrate"
	ModuleImageDecode    recipe.ModuleID = "model.image-decode"
	ModuleVQAPrepare     recipe.ModuleID = "model.vqa-prepare"
	ModuleVQAGenerate    recipe.ModuleID = "model.vqa-generate"
)

var catalog = mustCatalog()

type Plan struct {
	Identity  ProgramIdentity
	Recipe    recipe.Definition
	Model     model.ModelPlan
	Decode    DecodePlan
	Residency recipe.ResidencyPolicy
	Nodes     []recipe.Node
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
	Residency     recipe.ResidencyPolicy
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

func InferenceWithModelDefinition(
	modelID artifact.ID,
	profileID artifact.ID,
	definitionID artifact.ID,
	placement recipe.Placement,
	session DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
) (recipe.Definition, error) {
	return inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: profileID},
		{Role: recipe.DependencyDefinition, Artifact: definitionID},
	}, placement, session, residency)
}

type scalarStage struct {
	node               recipe.NodeID
	module             recipe.ModuleID
	input, output      recipe.PortName
	inputData, outData recipe.DataKind
}

type linearCapability struct {
	placement recipe.Placement
	stages    []scalarStage
}

func (c linearCapability) definition(task recipe.Task, modelID artifact.ID) (recipe.Definition, error) {
	nodes := make([]recipe.Node, len(c.stages))
	edges := make([]recipe.Edge, len(c.stages)-1)
	for index, stage := range c.stages {
		nodes[index] = recipe.Node{ID: stage.node, Module: stage.module, Placement: c.placement}
		if index > 0 {
			prior := c.stages[index-1]
			edges[index-1] = recipe.Edge{
				From: recipe.Endpoint{Node: prior.node, Port: prior.output},
				To:   recipe.Endpoint{Node: stage.node, Port: stage.input},
			}
		}
	}
	first, last := c.stages[0], c.stages[len(c.stages)-1]
	return recipe.NewDefinitionWithDependencies(
		task, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, nodes, edges,
		[]recipe.Input{{Name: first.input, Data: first.inputData, Target: recipe.Endpoint{Node: first.node, Port: first.input}}},
		[]recipe.Output{{Name: last.output, Data: last.outData, Source: recipe.Endpoint{Node: last.node, Port: last.output}}},
	)
}

func (c linearCapability) modules(task recipe.Task) []recipe.Module {
	modules := make([]recipe.Module, len(c.stages))
	for index, stage := range c.stages {
		placements := []recipe.Placement{c.placement}
		if task == recipe.TaskImageGen {
			placements = append(placements, recipe.PlacementHybrid)
		}
		modules[index] = recipe.Module{
			ID: stage.module, Tasks: []recipe.Task{task}, Placements: placements,
			Inputs:  []recipe.Port{{Name: stage.input, Data: stage.inputData, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: stage.output, Data: stage.outData, Cardinality: recipe.CardinalityOne}},
		}
	}
	return modules
}

var linearCapabilities = map[recipe.Task]linearCapability{
	recipe.TaskForecast: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "forecast", module: ModuleForecastSeries, input: "series", output: "forecast", inputData: recipe.DataTensor, outData: recipe.DataTensor},
	}},
	recipe.TaskTabular: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "tabular", module: ModuleTabularPredict, input: "table", output: "predictions", inputData: recipe.DataTensor, outData: recipe.DataTensor},
	}},
	recipe.TaskSeq2Seq: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "encode", module: ModuleSeq2SeqEncode, input: "source", output: "memory", inputData: recipe.DataTokens, outData: recipe.DataTensor},
		{node: "prepare", module: ModuleSeq2SeqPrepare, input: "memory", output: "session", inputData: recipe.DataTensor, outData: recipe.DataSessionPlan},
		{node: "select", module: ModuleSeq2SeqSelect, input: "session", output: "tokens", inputData: recipe.DataSessionPlan, outData: recipe.DataTokens},
	}},
	recipe.TaskSpeech: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "tokenize", module: ModuleSpeechTokenize, input: "text", output: "tokens", inputData: recipe.DataText, outData: recipe.DataTokens},
		{node: "generate", module: ModuleSpeechGenerate, input: "tokens", output: "latents", inputData: recipe.DataTokens, outData: recipe.DataTensor},
		{node: "decode", module: ModuleSpeechDecode, input: "latents", output: "audio", inputData: recipe.DataTensor, outData: recipe.DataAudio},
	}},
	recipe.TaskImageGen: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "prepare", module: ModuleImagePrepare, input: "condition", output: "session", inputData: recipe.DataTensor, outData: recipe.DataSessionPlan},
		{node: "integrate", module: ModuleImageIntegrate, input: "session", output: "features", inputData: recipe.DataSessionPlan, outData: recipe.DataTensor},
		{node: "decode", module: ModuleImageDecode, input: "features", output: "image", inputData: recipe.DataTensor, outData: recipe.DataImage},
	}},
}

// CapabilityDefinition: task-indexed executable topology.
func CapabilityDefinition(task recipe.Task, modelID artifact.ID) (recipe.Definition, error) {
	if task == recipe.TaskVQA {
		return vqaDefinition(modelID)
	}
	capability, ok := linearCapabilities[task]
	if !ok {
		return recipe.Definition{}, fmt.Errorf("model recipe: unsupported capability task %q", task)
	}
	return capability.definition(task, modelID)
}

// CapabilityDefinitionAt: executable topology with explicit placement.
func CapabilityDefinitionAt(task recipe.Task, modelID artifact.ID, placement recipe.Placement) (recipe.Definition, error) {
	capability, ok := linearCapabilities[task]
	if !ok || task == recipe.TaskVQA {
		return recipe.Definition{}, fmt.Errorf("model recipe: unsupported placed capability task %q", task)
	}
	if placement != capability.placement && (task != recipe.TaskImageGen || placement != recipe.PlacementHybrid) {
		return recipe.Definition{}, fmt.Errorf("model recipe: task %q does not support placement %q", task, placement)
	}
	capability.placement = placement
	return capability.definition(task, modelID)
}

func vqaDefinition(modelID artifact.ID) (recipe.Definition, error) {
	prepare := recipe.Node{ID: "prepare", Module: ModuleVQAPrepare, Placement: recipe.PlacementHost}
	generate := recipe.Node{ID: "generate", Module: ModuleVQAGenerate, Placement: recipe.PlacementDevice}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskVQA,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{prepare, generate},
		[]recipe.Edge{{
			From: recipe.Endpoint{Node: prepare.ID, Port: "session"},
			To:   recipe.Endpoint{Node: generate.ID, Port: "session"},
		}},
		[]recipe.Input{
			{
				Name: "image", Data: recipe.DataImage,
				Target: recipe.Endpoint{Node: prepare.ID, Port: "image"},
			},
			{
				Name: "question", Data: recipe.DataText,
				Target: recipe.Endpoint{Node: prepare.ID, Port: "question"},
			},
		},
		[]recipe.Output{{
			Name: "answer", Data: recipe.DataText,
			Source: recipe.Endpoint{Node: generate.ID, Port: "answer"},
		}},
	)
}

func inference(
	dependencies []recipe.Dependency,
	placement recipe.Placement,
	session DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
) (recipe.Definition, error) {
	if err := validateResidencyPlacement(residency, placement); err != nil {
		return recipe.Definition{}, err
	}
	compile := recipe.Node{
		ID: "compile", Module: ModuleCompileModelPlan, Placement: placement, Residency: residency,
	}
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

// CompileCapability resolves an executable non-inference model program.
func CompileCapability(definition recipe.Definition) (recipe.Program, error) {
	if definition.Task == recipe.TaskInference {
		return recipe.Program{}, fmt.Errorf("model recipe: unsupported runtime task %q", definition.Task)
	}
	return recipe.CompileProgram(definition, catalog)
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
	decode, residency, err := compileRuntimePolicies(
		definition.Nodes, modelPlan, programIdentity(definition).Placement,
	)
	if err != nil {
		return Plan{}, err
	}
	return compilePlan(definition, definition.Nodes, modelPlan, decode, residency), nil
}

func compilePlan(
	definition recipe.Definition,
	nodes []recipe.Node,
	modelPlan model.ModelPlan,
	decode DecodePlan,
	residency recipe.ResidencyPolicy,
) Plan {
	return Plan{
		Identity: programIdentity(definition), Recipe: definition,
		Model: modelPlan, Decode: decode, Residency: residency,
		Nodes: append([]recipe.Node(nil), nodes...),
	}
}

func compileRuntimePolicies(
	nodes []recipe.Node,
	modelPlan model.ModelPlan,
	placement recipe.Placement,
) (DecodePlan, recipe.ResidencyPolicy, error) {
	var session DecodeSessionPolicy
	var residency recipe.ResidencyPolicy
	for _, node := range nodes {
		if node.Session != "" {
			if node.Module != ModuleCompileDecodePlan || session != "" {
				return DecodePlan{}, "", errors.New("model recipe: misplaced or duplicate session policy")
			}
			session = node.Session
		}
		if node.Residency != "" {
			if node.Module != ModuleCompileModelPlan || residency != "" {
				return DecodePlan{}, "", errors.New("model recipe: misplaced or duplicate residency policy")
			}
			residency = node.Residency
		}
	}
	if session == "" || residency == "" {
		return DecodePlan{}, "", errors.New("model recipe: runtime policy is incomplete")
	}
	if session == DecodeSessionCapacity && !modelPlan.SupportsCapacityCache() {
		return DecodePlan{}, "", errors.New("model recipe: capacity session is incompatible with model plan")
	}
	if err := validateResidencyPlacement(residency, placement); err != nil {
		return DecodePlan{}, "", err
	}
	return DecodePlan{Session: session}, residency, nil
}

func validateResidencyPlacement(policy recipe.ResidencyPolicy, placement recipe.Placement) error {
	if policy == "" || !policy.Valid() {
		return errors.New("model recipe: residency policy is invalid")
	}
	if placement == recipe.PlacementDevice &&
		(policy == recipe.ResidencyStream || policy == recipe.ResidencyHostCache ||
			policy == recipe.ResidencyHybridNative || policy == recipe.ResidencyHostReference) {
		return errors.New("model recipe: residency policy is incompatible with device placement")
	}
	if placement == recipe.PlacementHost && policy != recipe.ResidencyHostReference && policy != recipe.ResidencyHostCache {
		return errors.New("model recipe: residency policy is incompatible with host placement")
	}
	return nil
}

func programIdentity(definition recipe.Definition) ProgramIdentity {
	identity := ProgramIdentity{
		Model: definition.Model, Recipe: definition.ID, RecipeVersion: definition.Version,
		Runtime: RuntimeInference,
	}
	identity.Profile, _ = definition.Dependency(recipe.DependencyProfile, 0)
	identity.Definition, _ = definition.Dependency(recipe.DependencyDefinition, 0)
	for _, node := range definition.Nodes {
		if node.Module == ModuleCompileModelPlan {
			identity.Residency = node.Residency
		}
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
		identity.Runtime != RuntimeInference || identity.Placement == "" || identity.Residency == "" {
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
	decode, residency, err := compileRuntimePolicies(p.Nodes, p.Model, p.Identity.Placement)
	if err != nil || decode != p.Decode || residency != p.Residency || residency != p.Identity.Residency {
		return errors.New("model recipe: serving runtime program differs")
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
	modules := []recipe.Module{
		{
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
	}
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen,
	} {
		modules = append(modules, linearCapabilities[task].modules(task)...)
	}
	modules = append(modules,
		recipe.Module{
			ID: ModuleVQAPrepare, Tasks: []recipe.Task{recipe.TaskVQA},
			Placements: []recipe.Placement{recipe.PlacementHost},
			Inputs: []recipe.Port{
				{Name: "image", Data: recipe.DataImage, Cardinality: recipe.CardinalityOne},
				{Name: "question", Data: recipe.DataText, Cardinality: recipe.CardinalityOne},
			},
			Outputs: []recipe.Port{{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleVQAGenerate, Tasks: []recipe.Task{recipe.TaskVQA},
			Placements: []recipe.Placement{recipe.PlacementDevice},
			Inputs:     []recipe.Port{{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne}},
			Outputs:    []recipe.Port{{Name: "answer", Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
		},
	)
	catalog, err := recipe.NewCatalog(modules...)
	if err != nil {
		panic(err)
	}
	return catalog
}
