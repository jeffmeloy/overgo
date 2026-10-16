package modelrecipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
)

const (
	// AudioContractMediaType identifies artifact-derived audio geometry and state.
	AudioContractMediaType = "application/vnd.overgo.audio-contract+json"
	// AudioContractSchema identifies the first audio contract document schema.
	AudioContractSchema = "overgo/audio-contract/v1"

	// ModuleTranscribeAudio identifies neutral audio-to-text execution.
	ModuleTranscribeAudio recipe.ModuleID = "model.transcribe-audio"
	// ModuleTranscribeSegments traverses declared activity intervals with the
	// transcription model, preserving their original sample coordinates.
	ModuleTranscribeSegments recipe.ModuleID = "model.transcribe-audio-segments"
	// ModuleAlignAudio identifies sample-exact transcript alignment.
	ModuleAlignAudio recipe.ModuleID = "model.align-audio"
	// ModuleDiarizeAudio identifies speaker-turn extraction.
	ModuleDiarizeAudio recipe.ModuleID = "model.diarize-audio"
	// ModuleDetectActivity identifies active-speech segmentation.
	ModuleDetectActivity recipe.ModuleID = "model.detect-audio-activity"
	// ModuleConvertAudio identifies content-preserving audio conversion.
	ModuleConvertAudio recipe.ModuleID = "model.convert-audio"
	// ModuleGenerateAudio identifies condition-to-audio generation.
	ModuleGenerateAudio recipe.ModuleID = "model.generate-audio"
)

// AudioContractDocument binds sample and frame geometry to optional context
// and restart-state artifacts. Recipes carry this profile identity instead of
// repeating format constants at call sites.
type AudioContractDocument struct {
	Version uint16                            `json:"version"`
	Format  recipecontract.AudioFormat        `json:"format"`
	Frame   recipecontract.AudioFrameGeometry `json:"frame"`
	Context artifact.ID                       `json:"context,omitzero"`
	State   artifact.ID                       `json:"state,omitzero"`
	ID      artifact.ID                       `json:"-"`
}

var audioContractCodec = artifact.JSONDocumentCodec(
	"audio contract", artifact.KindProfile, AudioContractMediaType, AudioContractSchema,
	canonicalizeAudioContract,
	func(value AudioContractDocument) artifact.ID { return value.ID },
	func(value *AudioContractDocument, id artifact.ID) { value.ID = id },
)

// NewAudioContract identifies one immutable decoded-audio execution contract.
func NewAudioContract(format recipecontract.AudioFormat, frame recipecontract.AudioFrameGeometry, contextID, stateID artifact.ID) (AudioContractDocument, error) {
	return audioContractCodec.NewInitial(AudioContractDocument{
		Format: format, Frame: frame, Context: contextID, State: stateID,
	})
}

// RequireAudioContract loads one exact decoded-audio execution contract.
func RequireAudioContract(ctx context.Context, reader artifact.Reader, id artifact.ID) (AudioContractDocument, error) {
	return audioContractCodec.RequireExactLineage(ctx, reader, id, AudioContractDocument.Lineage)
}

// ValidateIdentity verifies the contract's content-derived profile identity.
func (document AudioContractDocument) ValidateIdentity() error {
	return audioContractCodec.ValidateIdentity(document)
}

// Content returns the canonical contract artifact.
func (document AudioContractDocument) Content() (artifact.Content, error) {
	return audioContractCodec.Content(document)
}

// Lineage derives the exact context and state authorities used by the contract.
func (document AudioContractDocument) Lineage() []artifact.Lineage {
	var parents []artifact.ID
	if document.Context.Valid() {
		parents = append(parents, document.Context)
	}
	if document.State.Valid() {
		parents = append(parents, document.State)
	}
	return artifact.DependencyLineage(document.ID, parents...)
}

// Batch returns one atomic contract publication with exact dependency lineage.
func (document AudioContractDocument) Batch(key string) (artifact.Batch, error) {
	return audioContractCodec.Batch(key, document, document.Lineage(), nil)
}

func canonicalizeAudioContract(document *AudioContractDocument) error {
	if document == nil || document.Version != artifact.InitialDocumentVersion {
		return errors.New("model recipe: invalid audio contract envelope")
	}
	if err := document.Format.Validate(); err != nil {
		return err
	}
	if err := document.Frame.Validate(); err != nil {
		return err
	}
	if document.Context.Valid() && document.Context.Kind() != artifact.KindProfile {
		return errors.New("model recipe: audio context must be a profile artifact")
	}
	if document.State.Valid() && document.State.Kind() != artifact.KindCheckpoint {
		return errors.New("model recipe: audio state must be a checkpoint artifact")
	}
	return nil
}

type audioTaskSpec struct {
	module recipe.ModuleID
	inputs []recipe.Port
	output recipe.Port
}

var audioTaskSpecs = map[recipe.Task]audioTaskSpec{
	recipe.TaskTranscription: {
		module: ModuleTranscribeAudio,
		inputs: []recipe.Port{{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne}},
		output: recipe.Port{Name: "transcription", Data: recipe.DataTranscription, Cardinality: recipe.CardinalityOne},
	},
	recipe.TaskAlignment: {
		module: ModuleAlignAudio,
		inputs: []recipe.Port{
			{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne},
			{Name: "transcription", Data: recipe.DataTranscription, Cardinality: recipe.CardinalityOne},
		},
		output: recipe.Port{Name: "alignment", Data: recipe.DataTimestampedAlignment, Cardinality: recipe.CardinalityOne},
	},
	recipe.TaskDiarization: {
		module: ModuleDiarizeAudio,
		inputs: []recipe.Port{{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne}},
		output: recipe.Port{Name: "turns", Data: recipe.DataSpeechTurns, Cardinality: recipe.CardinalityOne},
	},
	recipe.TaskActivityDetection: {
		module: ModuleDetectActivity,
		inputs: []recipe.Port{{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne}},
		output: recipe.Port{Name: "segments", Data: recipe.DataActivitySegments, Cardinality: recipe.CardinalityOne},
	},
	recipe.TaskAudioConversion: {
		module: ModuleConvertAudio,
		inputs: []recipe.Port{{Name: "audio", Data: recipe.DataAudio, Cardinality: recipe.CardinalityOne}},
		output: recipe.Port{Name: "audio", Data: recipe.DataConvertedAudio, Cardinality: recipe.CardinalityOne},
	},
	recipe.TaskAudioGeneration: {
		module: ModuleGenerateAudio,
		inputs: []recipe.Port{{Name: "condition", Data: recipe.DataText, Cardinality: recipe.CardinalityOne}},
		output: recipe.Port{Name: "audio", Data: recipe.DataGeneratedAudio, Cardinality: recipe.CardinalityOne},
	},
}

func audioTaskModules() []recipe.Module {
	modules := make([]recipe.Module, 0, len(audioTaskSpecs))
	for task, spec := range audioTaskSpecs {
		tasks := []recipe.Task{task}
		placements := []recipe.Placement{recipe.PlacementHost}
		if task == recipe.TaskTranscription {
			placements = append(placements, recipe.PlacementHybrid)
		}
		if task == recipe.TaskActivityDetection {
			tasks = append(tasks, recipe.TaskTranscription)
		}
		modules = append(modules, recipe.Module{
			ID: spec.module, Tasks: tasks, Placements: placements,
			Inputs: spec.inputs, Outputs: []recipe.Port{spec.output},
		})
	}
	modules = append(modules, recipe.Module{
		ID: ModuleTranscribeSegments, Tasks: []recipe.Task{recipe.TaskTranscription}, Placements: []recipe.Placement{recipe.PlacementHost},
		Inputs:  []recipe.Port{{Name: "segments", Data: recipe.DataActivitySegments, Cardinality: recipe.CardinalityOne}},
		Outputs: []recipe.Port{audioTaskSpecs[recipe.TaskTranscription].output},
	})
	return modules
}

// TranscriptionDefinition binds generic host transcription to its exact model,
// decoded-audio contract, processor declaration, tokenizer, and tensor catalog.
func TranscriptionDefinition(modelID, contractID, processorID, tokenizerID, tensorInventoryID artifact.ID) (recipe.Definition, error) {
	return transcriptionDefinition(modelID, contractID, processorID, tokenizerID, tensorInventoryID, artifact.ID{})
}

// DeviceTranscriptionDefinition binds the canonical base to resident feed-forward execution.
// The host definition remains independently executable without a device.
func DeviceTranscriptionDefinition(base recipe.Definition) (recipe.Definition, error) {
	if err := base.ValidateIdentity(); err != nil {
		return recipe.Definition{}, err
	}
	model, _ := base.PrimaryDependency(recipe.DependencyModel)
	contract, _ := base.PrimaryDependency(recipe.DependencyProfile)
	processor, _ := base.PrimaryDependency(recipe.DependencyProcessorProfile)
	tokenizer, _ := base.PrimaryDependency(recipe.DependencyTokenizer)
	inventory, _ := base.PrimaryDependency(recipe.DependencyTensorInventory)
	expected, err := TranscriptionDefinition(model, contract, processor, tokenizer, inventory)
	if err != nil || expected.ID != base.ID {
		return recipe.Definition{}, errors.New("device transcription: canonical unadapted host base required")
	}
	nodes := slices.Clone(base.Nodes)
	nodes[0].Placement, nodes[0].Residency = recipe.PlacementHybrid, recipe.ResidencyHybridNative
	return recipe.NewDefinitionWithDependencies(base.Task, base.Dependencies, nodes, base.Edges, base.Inputs, base.Outputs)
}

// transcriptionTokenizerFile is the tokenizer a speech recognizer reads.
const transcriptionTokenizerFile = "tokenizer.json"

// TranscriptionTokenizer is the tokenizer a speech checkpoint's manifest binds
// as tokenizer.json, the one its recognizer reads.
func TranscriptionTokenizer(manifest artifact.Manifest) (artifact.ID, bool) {
	for _, component := range manifest.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == transcriptionTokenizerFile {
			return component.Artifact, true
		}
	}
	return artifact.ID{}, false
}

// CompileTranscription compiles a speech checkpoint into the transcription
// recipe the speech runtime loads: the model, tokenizer and tensor inventory
// of its Hugging Face inventory, the execution profile that declares how it
// runs, and the audio contract that profile's frontend reads at its sample
// rate and frame geometry. It returns the recipe and the contract, which
// publishes with the profile beside the inventory.
func CompileTranscription(inventory modelartifact.Inventory, profile artifact.ID, sampleRate uint64, frame recipecontract.AudioFrameGeometry) (recipe.Definition, AudioContractDocument, error) {
	tokenizer, found := TranscriptionTokenizer(inventory.Manifest)
	if !found {
		return recipe.Definition{}, AudioContractDocument{}, errors.New("model recipe: the speech checkpoint binds no tokenizer.json")
	}
	contract, err := NewAudioContract(recipecontract.AudioFormat{SampleRate: sampleRate, Channels: 1, Encoding: "pcm-f32le"}, frame, artifact.ID{}, artifact.ID{})
	if err != nil {
		return recipe.Definition{}, AudioContractDocument{}, err
	}
	definition, err := TranscriptionDefinition(inventory.Manifest.ID, contract.ID, profile, tokenizer, inventory.TensorInventory.ID)
	return definition, contract, err
}

// ActivityDefinition binds speech activity execution to its exact model,
// processor declaration and tensor inventory. The processor owns the declared
// frontend and offline or causal boundary policy; this topology is shared.
func ActivityDefinition(modelID, processorID, inventoryID artifact.ID) (recipe.Definition, error) {
	return speechAnalysisDefinition(recipe.TaskActivityDetection, "activity", modelID, processorID, inventoryID)
}

// DiarizationDefinition binds speaker-turn extraction to its exact model,
// operation profile and tensor inventory without transcription dependencies.
func DiarizationDefinition(modelID, processorID, inventoryID artifact.ID) (recipe.Definition, error) {
	return speechAnalysisDefinition(recipe.TaskDiarization, "diarize", modelID, processorID, inventoryID)
}

func speechAnalysisDefinition(task recipe.Task, nodeID recipe.NodeID, modelID, processorID, inventoryID artifact.ID) (recipe.Definition, error) {
	spec := audioTaskSpecs[task]
	node := recipe.Node{ID: nodeID, Module: spec.module, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity, Residency: recipe.ResidencyHostCache}
	return recipe.NewDefinitionWithDependencies(task,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}, {Role: recipe.DependencyProcessorProfile, Artifact: processorID}, {Role: recipe.DependencyTensorInventory, Artifact: inventoryID}},
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: "audio", Data: recipe.DataAudio, Target: recipe.Endpoint{Node: node.ID, Port: "audio"}}},
		[]recipe.Output{{Name: spec.output.Name, Data: spec.output.Data, Source: recipe.Endpoint{Node: node.ID, Port: spec.output.Name}}})
}

// AdaptedTranscriptionDefinition adds one exact checkpoint to the canonical
// base transcription topology. It does not activate or promote the recipe.
func AdaptedTranscriptionDefinition(base recipe.Definition, checkpoint artifact.ID) (recipe.Definition, error) {
	if err := base.ValidateIdentity(); err != nil {
		return recipe.Definition{}, err
	}
	roles := [...]recipe.DependencyRole{recipe.DependencyModel, recipe.DependencyProfile, recipe.DependencyProcessorProfile, recipe.DependencyTokenizer, recipe.DependencyTensorInventory}
	var ids [len(roles)]artifact.ID
	for index, role := range roles {
		var found bool
		ids[index], found = base.PrimaryDependency(role)
		if !found {
			return recipe.Definition{}, errors.New("adapted transcription: base dependency is absent")
		}
	}
	expected, err := TranscriptionDefinition(ids[0], ids[1], ids[2], ids[3], ids[4])
	if err != nil || expected.ID != base.ID || checkpoint.Kind() != artifact.KindCheckpoint {
		return recipe.Definition{}, errors.New("adapted transcription: base topology or checkpoint differs")
	}
	return transcriptionDefinition(ids[0], ids[1], ids[2], ids[3], ids[4], checkpoint)
}

func transcriptionDefinition(modelID, contractID, processorID, tokenizerID, tensorInventoryID, checkpoint artifact.ID) (recipe.Definition, error) {
	node := recipe.Node{
		ID: "transcribe", Module: ModuleTranscribeAudio, Placement: recipe.PlacementHost,
		Session: recipe.SessionCapacity, Residency: recipe.ResidencyHostCache,
	}
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: contractID},
		{Role: recipe.DependencyProcessorProfile, Artifact: processorID},
		{Role: recipe.DependencyTokenizer, Artifact: tokenizerID},
		{Role: recipe.DependencyTensorInventory, Artifact: tensorInventoryID},
	}
	if checkpoint.Valid() {
		dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyCheckpoint, Artifact: checkpoint})
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTranscription, dependencies,
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: "audio", Data: recipe.DataAudio, Target: recipe.Endpoint{Node: node.ID, Port: "audio"}}},
		[]recipe.Output{{Name: "transcription", Data: recipe.DataTranscription, Source: recipe.Endpoint{Node: node.ID, Port: "transcription"}}},
	)
}
