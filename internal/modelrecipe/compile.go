package modelrecipe

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/workflowrecipe"
)

const (
	ModuleCompileModelPlan  recipe.ModuleID = "model.compile-plan"
	ModuleCompileDecodePlan recipe.ModuleID = "model.compile-decode-plan"
	ModuleForwardTokens     recipe.ModuleID = "model.forward-tokens"
	// ModuleCaptureRepresentation identifies the model-boundary representation capture stage.
	ModuleCaptureRepresentation recipe.ModuleID = "model.capture-representation"
	// ModuleInjectRepresentation identifies the model-boundary representation injection stage.
	ModuleInjectRepresentation recipe.ModuleID = "model.inject-representation"
	ModuleThoughtBankGenerate  recipe.ModuleID = "model.thoughtbank-generate"
	// ModuleForecastSeries: host forward for series-forecast capability
	// packages; input series tensor, output quantile-forecast tensor.
	ModuleForecastSeries recipe.ModuleID = "model.forecast-series"
	// ModuleTabularPredict: host forward for tabular ICL capability
	// packages; input table tensor, output per-row predictions tensor.
	ModuleTabularPredict           recipe.ModuleID = "model.tabular-predict"
	ModuleSeq2SeqEncode            recipe.ModuleID = "model.seq2seq-encode"
	ModuleSeq2SeqPrepare           recipe.ModuleID = "model.seq2seq-prepare"
	ModuleSeq2SeqSelect            recipe.ModuleID = "model.seq2seq-select"
	ModuleSpeechTokenize           recipe.ModuleID = "model.speech-tokenize"
	ModuleSpeechGenerate           recipe.ModuleID = "model.speech-generate"
	ModuleSpeechDecode             recipe.ModuleID = "model.speech-decode"
	ModuleLatentImagePrepare       recipe.ModuleID = "model.latent-image-prepare"
	ModuleLatentImageIntegrate     recipe.ModuleID = "model.latent-image-integrate"
	ModuleLatentImageDecode        recipe.ModuleID = "model.latent-image-decode"
	ModuleOscillatorImagePrepare   recipe.ModuleID = "model.oscillator-image-prepare"
	ModuleOscillatorImageIntegrate recipe.ModuleID = "model.oscillator-image-integrate"
	ModuleOscillatorImageDecode    recipe.ModuleID = "model.oscillator-image-decode"
	ModuleDiffusionImagePrepare    recipe.ModuleID = "model.diffusion-image-prepare"
	ModuleDiffusionImageIntegrate  recipe.ModuleID = "model.diffusion-image-integrate"
	ModuleDiffusionImageDecode     recipe.ModuleID = "model.diffusion-image-decode"
	ModuleOscillatorVideoPrepare   recipe.ModuleID = "model.oscillator-video-prepare"
	ModuleOscillatorVideoIntegrate recipe.ModuleID = "model.oscillator-video-integrate"
	ModuleOscillatorVideoDecode    recipe.ModuleID = "model.oscillator-video-decode"
	ModuleRoutedImagePrepare       recipe.ModuleID = "model.routed-image-prepare"
	ModuleRoutedImageIntegrate     recipe.ModuleID = "model.routed-image-integrate"
	ModuleRoutedImageDecode        recipe.ModuleID = "model.routed-image-decode"
	ModuleLatentVideoPrepare       recipe.ModuleID = "model.latent-video-prepare"
	ModuleLatentVideoIntegrate     recipe.ModuleID = "model.latent-video-integrate"
	ModuleLatentVideoDecode        recipe.ModuleID = "model.latent-video-decode"
	ModuleReferenceVideoPrepare    recipe.ModuleID = "model.reference-video-prepare"
	ModuleReferenceVideoIntegrate  recipe.ModuleID = "model.reference-video-integrate"
	ModuleReferenceVideoDecode     recipe.ModuleID = "model.reference-video-decode"
	ModuleVQAPrepare               recipe.ModuleID = "model.vqa-prepare"
	ModuleVQAGenerate              recipe.ModuleID = "model.vqa-generate"
)

var catalog = mustCatalog()

type Plan struct {
	Identity  ProgramIdentity
	Recipe    recipe.Definition
	Model     model.ModelPlan
	Decode    DecodePlan
	Residency recipe.ResidencyPolicy
	Nodes     []recipe.Node
	Evidence  []artifact.ID
}

// Runtime defines compiled execution surface.
type Runtime string

const RuntimeInference Runtime = "inference"

// ProgramIdentity defines exact serving bindings.
type ProgramIdentity struct {
	Model         artifact.ID            `json:"model"`
	Profile       artifact.ID            `json:"profile"`
	Definition    artifact.ID            `json:"definition"`
	Recipe        artifact.ID            `json:"recipe"`
	RecipeVersion uint16                 `json:"recipe_version"`
	Placement     recipe.Placement       `json:"placement"`
	Residency     recipe.ResidencyPolicy `json:"residency"`
	Runtime       Runtime                `json:"runtime"`
}

// DecodeSessionPolicy defines compiled decode-graph lifetime.
type DecodeSessionPolicy = recipe.SessionPolicy

const (
	DecodeSessionRequest  = recipe.SessionRequest
	DecodeSessionCapacity = recipe.SessionCapacity
)

// DecodePlan defines recipe-owned session program.
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

const (
	sourceModelSlot uint32 = iota
	targetModelSlot
)

// RepresentationBridgeCompiler binds the execution and lifetime policies for
// two models that exchange an exact contracted representation through an
// adapter artifact.
type RepresentationBridgeCompiler struct {
	SourcePlacement recipe.Placement
	TargetPlacement recipe.Placement
	SourceResidency recipe.ResidencyPolicy
	TargetResidency recipe.ResidencyPolicy
	SourceSession   recipe.SessionPolicy
	TargetSession   recipe.SessionPolicy
}

// Definition compiles the source capture and target injection stages into one
// content-addressed recipe. Profile slots bind the representation contracts to
// the same source and target ordering as the model slots.
func (compiler RepresentationBridgeCompiler) Definition(
	sourceModel, targetModel artifact.ID,
	sourceContract, targetContract artifact.ID,
	bridge artifact.ID,
) (recipe.Definition, error) {
	if err := validateResidencyPlacement(compiler.SourceResidency, compiler.SourcePlacement); err != nil {
		return recipe.Definition{}, fmt.Errorf("model recipe: source representation component: %w", err)
	}
	if err := validateResidencyPlacement(compiler.TargetResidency, compiler.TargetPlacement); err != nil {
		return recipe.Definition{}, fmt.Errorf("model recipe: target representation component: %w", err)
	}
	if compiler.SourceSession == "" || !compiler.SourceSession.Valid() ||
		compiler.TargetSession == "" || !compiler.TargetSession.Valid() {
		return recipe.Definition{}, errors.New("model recipe: representation component session is invalid")
	}
	capture := recipe.Node{
		ID: "capture", Module: ModuleCaptureRepresentation,
		Placement: compiler.SourcePlacement, Session: compiler.SourceSession,
		Residency: compiler.SourceResidency, ModelSlot: sourceModelSlot,
	}
	inject := recipe.Node{
		ID: "inject", Module: ModuleInjectRepresentation,
		Placement: compiler.TargetPlacement, Session: compiler.TargetSession,
		Residency: compiler.TargetResidency, ModelSlot: targetModelSlot,
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskProjection,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Slot: sourceModelSlot, Artifact: sourceModel},
			{Role: recipe.DependencyModel, Slot: targetModelSlot, Artifact: targetModel},
			{Role: recipe.DependencyProfile, Slot: sourceModelSlot, Artifact: sourceContract},
			{Role: recipe.DependencyProfile, Slot: targetModelSlot, Artifact: targetContract},
			{Role: recipe.DependencyAdapter, Artifact: bridge},
		},
		[]recipe.Node{capture, inject},
		[]recipe.Edge{{
			From: recipe.Endpoint{Node: capture.ID, Port: "representation"},
			To:   recipe.Endpoint{Node: inject.ID, Port: "representation"},
		}},
		[]recipe.Input{{
			Name: "input", Data: recipe.DataTensor,
			Target: recipe.Endpoint{Node: capture.ID, Port: "input"},
		}},
		[]recipe.Output{{
			Name: "output", Data: recipe.DataTensor,
			Source: recipe.Endpoint{Node: inject.ID, Port: "output"},
		}},
	)
}

type scalarStage struct {
	node               recipe.NodeID
	module             recipe.ModuleID
	input, output      recipe.PortName
	inputData, outData recipe.DataKind
	session            recipe.SessionPolicy
}

type linearCapability struct {
	placement recipe.Placement
	stages    []scalarStage
}

type projectionStage struct {
	name    string
	media   recipe.DataKind
	tensor  recipe.DataKind
	decode  recipe.ModuleID
	project recipe.ModuleID
}

var projectionStages = map[recipe.DataKind]projectionStage{
	recipe.DataImage: {"image", recipe.DataImage, recipe.DataImageTensor, workflowrecipe.ModuleDecodeImage, workflowrecipe.ModuleProjectImage},
	recipe.DataAudio: {"audio", recipe.DataAudio, recipe.DataAudioTensor, workflowrecipe.ModuleDecodeAudio, workflowrecipe.ModuleProjectAudio},
	recipe.DataVideo: {"video", recipe.DataVideo, recipe.DataVideoTensor, workflowrecipe.ModuleDecodeVideo, workflowrecipe.ModuleProjectVideo},
}

// ProjectionDefinition returns one exact projector bundle with one branch per supported modality.
func ProjectionDefinition(
	modelID, projectorID, processorProfile artifact.ID,
	media ...recipe.DataKind,
) (recipe.Definition, error) {
	if len(media) == 0 {
		return recipe.Definition{}, errors.New("model recipe: projection bundle is empty")
	}
	media = slices.Clone(media)
	slices.Sort(media)
	media = slices.Compact(media)
	nodes := make([]recipe.Node, 0, 2*len(media))
	edges := make([]recipe.Edge, 0, len(media))
	inputs := make([]recipe.Input, 0, len(media))
	outputs := make([]recipe.Output, 0, len(media))
	for _, kind := range media {
		stage, ok := projectionStages[kind]
		if !ok {
			return recipe.Definition{}, fmt.Errorf("model recipe: unsupported projection media %q", kind)
		}
		decodeID := recipe.NodeID(stage.name + "-decode")
		projectID := recipe.NodeID(stage.name + "-project")
		nodes = append(nodes,
			recipe.Node{ID: decodeID, Module: stage.decode, Placement: recipe.PlacementHybrid},
			recipe.Node{ID: projectID, Module: stage.project, Placement: recipe.PlacementHybrid},
		)
		edges = append(edges, recipe.Edge{
			From: recipe.Endpoint{Node: decodeID, Port: "tensor"},
			To:   recipe.Endpoint{Node: projectID, Port: "tensor"},
		})
		inputs = append(inputs, recipe.Input{
			Name: recipe.PortName(stage.name), Data: stage.media,
			Target: recipe.Endpoint{Node: decodeID, Port: "media"},
		})
		outputs = append(outputs, recipe.Output{
			Name: recipe.PortName(stage.name + "-embeddings"), Data: recipe.DataEmbeddings,
			Source: recipe.Endpoint{Node: projectID, Port: "embeddings"},
		})
	}
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProjector, Artifact: projectorID},
	}
	if processorProfile.Valid() {
		dependencies = append(dependencies, recipe.Dependency{
			Role: recipe.DependencyProcessorProfile, Artifact: processorProfile,
		})
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskProjection,
		dependencies,
		nodes, edges, inputs, outputs,
	)
}

func (c linearCapability) definition(task recipe.Task, modelID artifact.ID, dependencies ...recipe.Dependency) (recipe.Definition, error) {
	nodes := make([]recipe.Node, len(c.stages))
	edges := make([]recipe.Edge, len(c.stages)-1)
	for index, stage := range c.stages {
		nodes[index] = recipe.Node{ID: stage.node, Module: stage.module, Placement: c.placement, Session: stage.session}
		if index > 0 {
			prior := c.stages[index-1]
			edges[index-1] = recipe.Edge{
				From: recipe.Endpoint{Node: prior.node, Port: prior.output},
				To:   recipe.Endpoint{Node: stage.node, Port: stage.input},
			}
		}
	}
	first, last := c.stages[0], c.stages[len(c.stages)-1]
	dependencies = append([]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}}, dependencies...)
	return recipe.NewDefinitionWithDependencies(
		task, dependencies, nodes, edges,
		[]recipe.Input{{Name: first.input, Data: first.inputData, Target: recipe.Endpoint{Node: first.node, Port: first.input}}},
		[]recipe.Output{{Name: last.output, Data: last.outData, Source: recipe.Endpoint{Node: last.node, Port: last.output}}},
	)
}

func (c linearCapability) modules(task recipe.Task) []recipe.Module {
	modules := make([]recipe.Module, len(c.stages))
	for index, stage := range c.stages {
		modules[index] = recipe.Module{
			ID: stage.module, Tasks: []recipe.Task{task}, Placements: []recipe.Placement{c.placement},
			Inputs:  []recipe.Port{{Name: stage.input, Data: stage.inputData, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: stage.output, Data: stage.outData, Cardinality: recipe.CardinalityOne}},
		}
	}
	return modules
}

var linearCapabilities = map[recipe.Task]linearCapability{
	recipe.TaskGeneration: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "generate", module: ModuleThoughtBankGenerate, input: "request", output: "text", inputData: recipe.DataText, outData: recipe.DataText, session: recipe.SessionCapacity},
	}},
	recipe.TaskForecast: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "forecast", module: ModuleForecastSeries, input: "series", output: "forecast", inputData: recipe.DataTensor, outData: recipe.DataTensor},
	}},
	recipe.TaskTabular: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "tabular", module: ModuleTabularPredict, input: "table", output: "predictions", inputData: recipe.DataTensor, outData: recipe.DataTensor},
	}},
	recipe.TaskSeq2Seq: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "encode", module: ModuleSeq2SeqEncode, input: "source", output: "memory", inputData: recipe.DataText, outData: recipe.DataTensor},
		{node: "prepare", module: ModuleSeq2SeqPrepare, input: "memory", output: "session", inputData: recipe.DataTensor, outData: recipe.DataSessionPlan},
		{node: "select", module: ModuleSeq2SeqSelect, input: "session", output: "text", inputData: recipe.DataSessionPlan, outData: recipe.DataText},
	}},
	recipe.TaskSpeech: {placement: recipe.PlacementHost, stages: []scalarStage{
		{node: "tokenize", module: ModuleSpeechTokenize, input: "text", output: "tokens", inputData: recipe.DataText, outData: recipe.DataTokens},
		{node: "generate", module: ModuleSpeechGenerate, input: "tokens", output: "latents", inputData: recipe.DataTokens, outData: recipe.DataTensor},
		{node: "decode", module: ModuleSpeechDecode, input: "latents", output: "audio", inputData: recipe.DataTensor, outData: recipe.DataAudio},
	}},
}

func imageCapability(
	placement recipe.Placement,
	condition recipe.DataKind,
	prepare, integrate, decode recipe.ModuleID,
) linearCapability {
	return linearCapability{placement: placement, stages: []scalarStage{
		{node: "prepare", module: prepare, input: "condition", output: "session", inputData: condition, outData: recipe.DataSessionPlan, session: recipe.SessionCapacity},
		{node: "integrate", module: integrate, input: "session", output: "features", inputData: recipe.DataSessionPlan, outData: recipe.DataTensor},
		{node: "decode", module: decode, input: "features", output: "image", inputData: recipe.DataTensor, outData: recipe.DataImage},
	}}
}

var latentImageCapability = imageCapability(recipe.PlacementHybrid, recipe.DataPromptConditioning, ModuleLatentImagePrepare, ModuleLatentImageIntegrate, ModuleLatentImageDecode)

var oscillatorImageCapability = imageCapability(recipe.PlacementHost, recipe.DataClassConditioning, ModuleOscillatorImagePrepare, ModuleOscillatorImageIntegrate, ModuleOscillatorImageDecode)

var diffusionImageCapability = imageCapability(recipe.PlacementHost, recipe.DataImageTensor, ModuleDiffusionImagePrepare, ModuleDiffusionImageIntegrate, ModuleDiffusionImageDecode)

var oscillatorVideoCapability = linearCapability{placement: recipe.PlacementHost, stages: []scalarStage{
	{node: "prepare", module: ModuleOscillatorVideoPrepare, input: "condition", output: "session", inputData: recipe.DataClassConditioning, outData: recipe.DataSessionPlan},
	{node: "integrate", module: ModuleOscillatorVideoIntegrate, input: "session", output: "features", inputData: recipe.DataSessionPlan, outData: recipe.DataVideoTensor},
	{node: "decode", module: ModuleOscillatorVideoDecode, input: "features", output: "video", inputData: recipe.DataVideoTensor, outData: recipe.DataVideo},
}}

var routedImageCapability = imageCapability(recipe.PlacementHybrid, recipe.DataPromptConditioning, ModuleRoutedImagePrepare, ModuleRoutedImageIntegrate, ModuleRoutedImageDecode)

var latentVideoCapability = linearCapability{placement: recipe.PlacementHybrid, stages: []scalarStage{
	{node: "prepare", module: ModuleLatentVideoPrepare, input: "condition", output: "session", inputData: recipe.DataPromptConditioning, outData: recipe.DataSessionPlan, session: recipe.SessionCapacity},
	{node: "integrate", module: ModuleLatentVideoIntegrate, input: "session", output: "features", inputData: recipe.DataSessionPlan, outData: recipe.DataVideoTensor},
	{node: "decode", module: ModuleLatentVideoDecode, input: "features", output: "video", inputData: recipe.DataVideoTensor, outData: recipe.DataVideo},
}}

// CapabilityDefinition returns task-indexed executable topology.
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

// LatentImageDefinition returns a prompt-conditioned diffusion image graph.
func LatentImageDefinition(modelID, profileID artifact.ID) (recipe.Definition, error) {
	return latentImageCapability.definition(recipe.TaskImageGen, modelID, recipe.Dependency{
		Role: recipe.DependencyProfile, Artifact: profileID,
	})
}

// OscillatorImageDefinition returns a class-conditioned oscillator image graph.
func OscillatorImageDefinition(modelID artifact.ID) (recipe.Definition, error) {
	return oscillatorImageCapability.definition(recipe.TaskImageGen, modelID)
}

// DiffusionImageDefinition returns a seeded image-tensor flow graph.
func DiffusionImageDefinition(modelID artifact.ID) (recipe.Definition, error) {
	return diffusionImageCapability.definition(recipe.TaskImageGen, modelID)
}

// OscillatorVideoDefinition returns a class-conditioned oscillator video graph.
func OscillatorVideoDefinition(modelID artifact.ID) (recipe.Definition, error) {
	return oscillatorVideoCapability.definition(recipe.TaskVideoGen, modelID)
}

// RoutedImageDefinition returns a prompt-conditioned routed-transformer image graph.
func RoutedImageDefinition(modelID artifact.ID) (recipe.Definition, error) {
	return routedImageCapability.definition(recipe.TaskImageGen, modelID)
}

// LatentVideoDefinition returns a prompt-conditioned latent-video graph.
func LatentVideoDefinition(modelID, profileID artifact.ID) (recipe.Definition, error) {
	return latentVideoCapability.definition(recipe.TaskVideoGen, modelID, recipe.Dependency{
		Role: recipe.DependencyProfile, Artifact: profileID,
	})
}

// ReferenceVideoEditDefinition returns a prompt and source-video conditioned graph.
func ReferenceVideoEditDefinition(modelID, profileID artifact.ID) (recipe.Definition, error) {
	prepare := recipe.Node{ID: "prepare", Module: ModuleReferenceVideoPrepare, Placement: recipe.PlacementHybrid, Session: recipe.SessionCapacity}
	integrate := recipe.Node{ID: "integrate", Module: ModuleReferenceVideoIntegrate, Placement: recipe.PlacementHybrid}
	decode := recipe.Node{ID: "decode", Module: ModuleReferenceVideoDecode, Placement: recipe.PlacementHybrid}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskVideoGen,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: modelID},
			{Role: recipe.DependencyProfile, Artifact: profileID},
		},
		[]recipe.Node{prepare, integrate, decode},
		[]recipe.Edge{
			{From: recipe.Endpoint{Node: prepare.ID, Port: "session"}, To: recipe.Endpoint{Node: integrate.ID, Port: "session"}},
			{From: recipe.Endpoint{Node: integrate.ID, Port: "features"}, To: recipe.Endpoint{Node: decode.ID, Port: "features"}},
		},
		[]recipe.Input{
			{Name: "condition", Data: recipe.DataPromptConditioning, Target: recipe.Endpoint{Node: prepare.ID, Port: "condition"}},
			{Name: "source", Data: recipe.DataVideo, Target: recipe.Endpoint{Node: prepare.ID, Port: "source"}},
		},
		[]recipe.Output{{Name: "video", Data: recipe.DataVideo, Source: recipe.Endpoint{Node: decode.ID, Port: "video"}}},
	)
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
	if definition.Task == recipe.TaskProjection || definition.Task == recipe.TaskTraining {
		return recipe.CompileProgram(definition, workflowrecipe.Catalog())
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
	identity.Profile, _ = definition.PrimaryDependency(recipe.DependencyProfile)
	identity.Definition, _ = definition.PrimaryDependency(recipe.DependencyDefinition)
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

// ValidateServing validates a complete identity-bound inference program.
func (p Plan) ValidateServing() error {
	if err := validateProgramIdentity(p.Recipe, p.Identity); err != nil {
		return err
	}
	if !p.Model.Compiled() || !p.Model.HasCacheSchemas() {
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
		{
			ID: ModuleCaptureRepresentation, Tasks: []recipe.Task{recipe.TaskProjection}, Placements: placements,
			Inputs:  []recipe.Port{{Name: "input", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: "representation", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
		},
		{
			ID: ModuleInjectRepresentation, Tasks: []recipe.Task{recipe.TaskProjection}, Placements: placements,
			Inputs:  []recipe.Port{{Name: "representation", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: "output", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
		},
	}
	for _, task := range []recipe.Task{
		recipe.TaskGeneration, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq, recipe.TaskSpeech,
	} {
		modules = append(modules, linearCapabilities[task].modules(task)...)
	}
	for _, capability := range []linearCapability{latentImageCapability, oscillatorImageCapability, diffusionImageCapability, routedImageCapability} {
		modules = append(modules, capability.modules(recipe.TaskImageGen)...)
	}
	modules = append(modules, oscillatorVideoCapability.modules(recipe.TaskVideoGen)...)
	modules = append(modules, latentVideoCapability.modules(recipe.TaskVideoGen)...)
	modules = append(modules,
		recipe.Module{
			ID: ModuleReferenceVideoPrepare, Tasks: []recipe.Task{recipe.TaskVideoGen}, Placements: []recipe.Placement{recipe.PlacementHybrid},
			Inputs: []recipe.Port{
				{Name: "condition", Data: recipe.DataPromptConditioning, Cardinality: recipe.CardinalityOne},
				{Name: "source", Data: recipe.DataVideo, Cardinality: recipe.CardinalityOne},
			},
			Outputs: []recipe.Port{{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleReferenceVideoIntegrate, Tasks: []recipe.Task{recipe.TaskVideoGen}, Placements: []recipe.Placement{recipe.PlacementHybrid},
			Inputs:  []recipe.Port{{Name: "session", Data: recipe.DataSessionPlan, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: "features", Data: recipe.DataVideoTensor, Cardinality: recipe.CardinalityOne}},
		},
		recipe.Module{
			ID: ModuleReferenceVideoDecode, Tasks: []recipe.Task{recipe.TaskVideoGen}, Placements: []recipe.Placement{recipe.PlacementHybrid},
			Inputs:  []recipe.Port{{Name: "features", Data: recipe.DataVideoTensor, Cardinality: recipe.CardinalityOne}},
			Outputs: []recipe.Port{{Name: "video", Data: recipe.DataVideo, Cardinality: recipe.CardinalityOne}},
		},
	)
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
