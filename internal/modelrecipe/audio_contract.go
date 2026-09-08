package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
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
	func(value AudioContractDocument) AudioContractDocument { return value },
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
		if task == recipe.TaskActivityDetection {
			tasks = append(tasks, recipe.TaskTranscription)
		}
		modules = append(modules, recipe.Module{
			ID: spec.module, Tasks: tasks, Placements: []recipe.Placement{recipe.PlacementHost},
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

// ActivityDefinition binds speech activity execution to its exact model,
// processor declaration and tensor inventory. The processor owns the declared
// frontend and offline or causal boundary policy; this topology is shared.
func ActivityDefinition(modelID, processorID, inventoryID artifact.ID) (recipe.Definition, error) {
	node := recipe.Node{ID: "activity", Module: ModuleDetectActivity, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity, Residency: recipe.ResidencyHostCache}
	return recipe.NewDefinitionWithDependencies(recipe.TaskActivityDetection,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}, {Role: recipe.DependencyProcessorProfile, Artifact: processorID}, {Role: recipe.DependencyTensorInventory, Artifact: inventoryID}},
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: "audio", Data: recipe.DataAudio, Target: recipe.Endpoint{Node: node.ID, Port: "audio"}}},
		[]recipe.Output{{Name: "segments", Data: recipe.DataActivitySegments, Source: recipe.Endpoint{Node: node.ID, Port: "segments"}}})
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
