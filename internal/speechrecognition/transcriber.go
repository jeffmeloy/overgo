package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/scratch"
	"overgo/internal/strictjson"
)

// TranscriptionSchema identifies persisted typed transcription output.
const TranscriptionSchema = "overgo/audio-transcription/v1"

// AudioAdmissionFailure is the stable run-record failure code for audio
// rejected before model execution.
const AudioAdmissionFailure = "audio-admission-refused"

// AudioFormatFailure identifies decoded audio incompatible with the recipe.
const AudioFormatFailure = "audio-format-mismatch"

var transcriptionContract = artifact.JSONContract(artifact.KindOutput, TranscriptionSchema)

// ErrAudioAdmissionRefused reports deterministic quarantine before inference.
var ErrAudioAdmissionRefused = errors.New("speech recognition: audio admission refused")

// RunBinding supplies the immutable execution authorities recorded with a run.
type RunBinding struct {
	Key         string
	CodeCommit  string
	Environment artifact.ID
	Dataset     artifact.ID
	Split       artifact.ID
}

// TranscriptionWorkspace owns all reusable mutable storage for one caller.
// Its zero value is ready to use and must not be shared concurrently.
type TranscriptionWorkspace struct {
	Frontend audiodsp.Workspace
	Encoder  Workspace
	frameIDs []int
	tokens   []int
}

// Transcriber owns immutable, validated CPU execution state. It is safe to
// share when each concurrent caller supplies a distinct workspace.
type Transcriber struct {
	repository artifact.Repository
	model      artifact.ID
	recipe     recipe.Definition
	contract   modelrecipe.AudioContractDocument
	profile    ExecutionProfile
	frontend   *audiodsp.Frontend
	encoder    *Encoder
	tokenizer  *hfbpe.Tokenizer
}

// LoadTranscriber resolves an exact stored transcription recipe and all of its
// content-addressed model dependencies once, before serving any clip.
func LoadTranscriber(ctx context.Context, repository artifact.Repository, definitionID artifact.ID, memoryBytes uint64) (*Transcriber, error) {
	if ctx == nil || repository == nil || definitionID.Kind() != artifact.KindRecipe || memoryBytes == 0 {
		return nil, errors.New("speech recognition: invalid transcriber load")
	}
	definition, err := recipe.RequireDefinition(ctx, repository, definitionID)
	if err != nil {
		return nil, err
	}
	if _, err = modelrecipe.CompileCapability(definition); err != nil {
		return nil, err
	}
	return loadTranscriber(ctx, repository, definition, memoryBytes)
}

// LoadActiveTranscriber resolves the verified active recipe for one model.
func LoadActiveTranscriber(ctx context.Context, repository artifact.Repository, modelID artifact.ID, memoryBytes uint64) (*Transcriber, error) {
	if ctx == nil || repository == nil {
		return nil, errors.New("speech recognition: invalid active transcriber load")
	}
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, repository, modelID, recipe.TaskTranscription)
	if err != nil {
		return nil, err
	}
	return loadTranscriber(ctx, repository, activation.Definition, memoryBytes)
}

func loadTranscriber(ctx context.Context, repository artifact.Repository, definition recipe.Definition, memoryBytes uint64) (*Transcriber, error) {
	if definition.Task != recipe.TaskTranscription {
		return nil, errors.New("speech recognition: recipe is not transcription")
	}
	modelID, contractID, profileID, tokenizerID, inventoryID, err := transcriptionDependencies(definition)
	if err != nil {
		return nil, err
	}
	expectedDefinition, err := modelrecipe.TranscriptionDefinition(modelID, contractID, profileID, tokenizerID, inventoryID)
	if err != nil || expectedDefinition.ID != definition.ID {
		return nil, errors.Join(errors.New("speech recognition: transcription recipe topology differs"), err)
	}
	if err = requireRecipeDependencyLineage(ctx, repository, definition); err != nil {
		return nil, err
	}
	contract, err := modelrecipe.RequireAudioContract(ctx, repository, contractID)
	if err != nil {
		return nil, err
	}
	expectedContract, err := modelrecipe.NewAudioContract(contract.Format, contract.Frame, contract.Context, contract.State)
	if err != nil || expectedContract.ID != contract.ID {
		return nil, errors.Join(errors.New("speech recognition: audio contract identity differs"), err)
	}
	profile, err := RequireExecutionProfile(ctx, repository, profileID)
	if err != nil {
		return nil, err
	}
	if contract.Format.SampleRate != uint64(profile.Frontend.SampleRate) || contract.Format.Channels != 1 ||
		contract.Format.Encoding != "pcm-f32le" || contract.Frame != profile.Frontend.Geometry {
		return nil, errors.New("speech recognition: audio contract differs from frontend declaration")
	}
	frontend, err := audiodsp.NewFrontend(profile.Frontend, memoryBytes)
	if err != nil {
		return nil, err
	}
	width, ok := checked.ProductInt(int(profile.Frontend.Geometry.FeatureBins), profile.Grouping.StackFrames)
	if profile.Grouping.DeltaRadius > 0 {
		width, ok = checked.MulInt(width, 2)
	}
	if !ok {
		return nil, errors.New("speech recognition: grouped feature width overflows")
	}
	modelPath, err := artifact.AvailablePath(ctx, repository, modelID, artifact.LocationDirectory)
	if err != nil {
		return nil, err
	}
	hf, err := hfrepo.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer hf.Close()
	inventory, err := modelartifact.FromHFRepository(hf)
	if err != nil {
		return nil, err
	}
	if inventory.Manifest.ID != modelID || inventory.TensorInventory.ID != inventoryID ||
		!manifestBindsTokenizer(inventory.Manifest, tokenizerID) {
		return nil, errors.New("speech recognition: model artifacts differ from recipe dependencies")
	}
	storedManifest, found, err := repository.Manifest(ctx, modelID)
	if err != nil || !found || storedManifest.ID != inventory.Manifest.ID || !slices.Equal(storedManifest.Components, inventory.Manifest.Components) {
		return nil, errors.Join(errors.New("speech recognition: stored model manifest differs"), err)
	}
	encoder, err := LoadEncoder(ctx, hf.Tensors, profile.Encoder, memoryBytes)
	if err != nil {
		return nil, err
	}
	if encoder.InputWidth() != width || profile.BlankToken >= encoder.VocabularySize() {
		return nil, errors.New("speech recognition: processor geometry differs from encoder")
	}
	tokenizer, err := hfbpe.Load(modelPath)
	if err != nil {
		return nil, err
	}
	return &Transcriber{
		repository: repository, model: modelID, recipe: definition, contract: contract, profile: profile,
		frontend: frontend, encoder: encoder, tokenizer: tokenizer,
	}, nil
}

// Transcribe admits, decodes, executes, and persists one bounded offline clip.
// The returned values are the existing public transcription and run contracts.
func (transcriber *Transcriber) Transcribe(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, workspace *TranscriptionWorkspace, binding RunBinding) (recipecontract.Transcription, runrecord.Run, error) {
	if transcriber == nil || ctx == nil || workspace == nil || binding.Key == "" ||
		binding.Dataset.Valid() != binding.Split.Valid() ||
		binding.Dataset.Valid() && (binding.Dataset.Kind() != artifact.KindDataset || binding.Split.Kind() != artifact.KindDatasetShard) {
		return recipecontract.Transcription{}, runrecord.Run{}, errors.New("speech recognition: invalid transcription execution")
	}
	started := time.Now()
	decodeStart := time.Now()
	inspection, err := dataset.InspectAudio(ctx, transcriber.repository, data, origin, policy)
	decodeDuration := elapsedNanoseconds(decodeStart)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	inputs := uniqueIDs(
		transcriber.model, binding.Dataset, binding.Split,
		inspection.Signal.Source.Audio, inspection.Signal.Source.Profile,
		inspection.SignalID, inspection.PolicyID, inspection.DecisionID,
	)
	if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
		run, runErr := transcriber.persistRun(ctx, binding, runrecord.OutcomeFailed, inputs, nil,
			AudioAdmissionFailure, elapsedNanoseconds(started), []runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: decodeDuration}})
		return recipecontract.Transcription{}, run, errors.Join(ErrAudioAdmissionRefused, runErr)
	}
	if inspection.Signal.Format != transcriber.contract.Format {
		run, runErr := transcriber.persistRun(ctx, binding, runrecord.OutcomeFailed, inputs, nil,
			AudioFormatFailure, elapsedNanoseconds(started), []runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: decodeDuration}})
		return recipecontract.Transcription{}, run, errors.Join(errors.New("speech recognition: decoded audio format differs"), runErr)
	}
	prepareStart := time.Now()
	features, frames, _, err := transcriber.frontend.ProcessGrouped(ctx, inspection.Samples, int(inspection.Signal.Format.SampleRate), &workspace.Frontend, transcriber.profile.Grouping)
	prepareDuration := elapsedNanoseconds(prepareStart)
	if err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-prepare-failed", started,
			[]runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: decodeDuration}, {Phase: runrecord.PhasePrepare, DurationNS: prepareDuration}}, err)
	}
	prefillStart := time.Now()
	_, logits, outputFrames, err := transcriber.encoder.Forward(ctx, features, frames, &workspace.Encoder, nil)
	prefillDuration := elapsedNanoseconds(prefillStart)
	if err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-inference-failed", started,
			[]runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: decodeDuration}, {Phase: runrecord.PhasePrepare, DurationNS: prepareDuration}, {Phase: runrecord.PhasePrefill, DurationNS: prefillDuration}}, err)
	}
	postStart := time.Now()
	workspace.frameIDs = scratch.Resize(workspace.frameIDs, outputFrames)
	workspace.tokens = scratch.Resize(workspace.tokens, outputFrames)
	tokens, err := GreedyCTC(ctx, workspace.frameIDs, workspace.tokens, logits, outputFrames, transcriber.encoder.VocabularySize(), transcriber.profile.BlankToken)
	if err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-postprocess-failed", started,
			completedTranscriptionPhases(decodeDuration, prepareDuration, prefillDuration, elapsedNanoseconds(postStart)), err)
	}
	text, err := transcriber.tokenizer.DecodeStrict(tokens)
	if err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-postprocess-failed", started,
			completedTranscriptionPhases(decodeDuration, prepareDuration, prefillDuration, elapsedNanoseconds(postStart)), err)
	}
	result := recipecontract.Transcription{Source: inspection.Signal.Source, Text: text, Language: transcriber.profile.Language}
	if err = result.Validate(); err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-postprocess-failed", started,
			completedTranscriptionPhases(decodeDuration, prepareDuration, prefillDuration, elapsedNanoseconds(postStart)), err)
	}
	output, err := artifact.JSONContent(transcriptionContract, result)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	postDuration := elapsedNanoseconds(postStart)
	phases := []runrecord.PhaseMetric{
		{Phase: runrecord.PhaseMediaDecode, DurationNS: decodeDuration},
		{Phase: runrecord.PhasePrepare, DurationNS: prepareDuration},
		{Phase: runrecord.PhasePrefill, DurationNS: prefillDuration},
		{Phase: runrecord.PhasePostprocess, DurationNS: postDuration},
	}
	run, err := runrecord.NewBoundRun(transcriber.recipe.ID, runrecord.OutcomeSucceeded, inputs,
		[]artifact.ID{output.Descriptor.ID}, "", binding.CodeCommit, binding.Environment,
		elapsedNanoseconds(started), phases)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	batch, err := artifact.NewDocumentBatch(binding.Key, []artifact.Content{output, runContent}, run.Lineage(), nil)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	if _, err = artifact.CommitBatch(ctx, transcriber.repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	return result, run, nil
}

// RequireTranscription loads one canonical persisted transcription output.
func RequireTranscription(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipecontract.Transcription, error) {
	var transcription recipecontract.Transcription
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return transcription, err
	}
	if !found {
		return transcription, errors.New("speech recognition: transcription output is absent")
	}
	if err = transcriptionContract.ValidateContent(content, id); err != nil {
		return transcription, err
	}
	if err = strictjson.DecodeBytes(content.Data, &transcription); err != nil {
		return recipecontract.Transcription{}, err
	}
	if err = transcription.Validate(); err != nil {
		return recipecontract.Transcription{}, err
	}
	canonical, err := artifact.JSONContent(transcriptionContract, transcription)
	if err != nil || canonical.Descriptor.ID != id {
		return recipecontract.Transcription{}, errors.Join(errors.New("speech recognition: transcription output is not canonical"), err)
	}
	return transcription, nil
}

func (transcriber *Transcriber) persistRun(ctx context.Context, binding RunBinding, outcome runrecord.Outcome, inputs, outputs []artifact.ID, failure string, measured uint64, phases []runrecord.PhaseMetric) (runrecord.Run, error) {
	run, err := runrecord.NewBoundRun(transcriber.recipe.ID, outcome, inputs, outputs, failure,
		binding.CodeCommit, binding.Environment, measured, phases)
	if err != nil {
		return runrecord.Run{}, err
	}
	batch, err := run.Batch(binding.Key)
	if err != nil {
		return runrecord.Run{}, err
	}
	if _, err = artifact.CommitBatch(ctx, transcriber.repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return runrecord.Run{}, err
	}
	return run, nil
}

func (transcriber *Transcriber) failedExecution(ctx context.Context, binding RunBinding, inputs []artifact.ID, failure string, started time.Time, phases []runrecord.PhaseMetric, executionErr error) (recipecontract.Transcription, runrecord.Run, error) {
	if ctx.Err() != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, executionErr
	}
	run, runErr := transcriber.persistRun(ctx, binding, runrecord.OutcomeFailed, inputs, nil,
		failure, elapsedNanoseconds(started), phases)
	return recipecontract.Transcription{}, run, errors.Join(executionErr, runErr)
}

func completedTranscriptionPhases(decode, prepare, prefill, postprocess uint64) []runrecord.PhaseMetric {
	return []runrecord.PhaseMetric{
		{Phase: runrecord.PhaseMediaDecode, DurationNS: decode},
		{Phase: runrecord.PhasePrepare, DurationNS: prepare},
		{Phase: runrecord.PhasePrefill, DurationNS: prefill},
		{Phase: runrecord.PhasePostprocess, DurationNS: postprocess},
	}
}

func transcriptionDependencies(definition recipe.Definition) (artifact.ID, artifact.ID, artifact.ID, artifact.ID, artifact.ID, error) {
	roles := [...]recipe.DependencyRole{recipe.DependencyModel, recipe.DependencyProfile, recipe.DependencyProcessorProfile, recipe.DependencyTokenizer, recipe.DependencyTensorInventory}
	ids := make([]artifact.ID, len(roles))
	for index, role := range roles {
		id, found := definition.PrimaryDependency(role)
		if !found {
			return artifact.ID{}, artifact.ID{}, artifact.ID{}, artifact.ID{}, artifact.ID{}, fmt.Errorf("speech recognition: recipe lacks %s dependency", role)
		}
		ids[index] = id
	}
	return ids[0], ids[1], ids[2], ids[3], ids[4], nil
}

func requireRecipeDependencyLineage(ctx context.Context, reader artifact.Reader, definition recipe.Definition) error {
	parents, err := reader.Parents(ctx, definition.ID)
	if err != nil {
		return err
	}
	for _, dependency := range definition.Dependencies {
		edge := artifact.Lineage{Child: definition.ID, Parent: dependency.Artifact, Relation: artifact.RelationDependsOn}
		if !slices.Contains(parents, edge) {
			return errors.New("speech recognition: stored recipe dependency lineage differs")
		}
		if _, found, artifactErr := reader.Artifact(ctx, dependency.Artifact); artifactErr != nil || !found {
			return errors.Join(errors.New("speech recognition: recipe dependency is absent"), artifactErr)
		}
	}
	return nil
}

func manifestBindsTokenizer(manifest artifact.Manifest, tokenizerID artifact.ID) bool {
	for _, component := range manifest.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == "tokenizer.json" && component.Artifact == tokenizerID {
			return true
		}
	}
	return false
}

func uniqueIDs(ids ...artifact.ID) []artifact.ID {
	result := make([]artifact.ID, 0, len(ids))
	for _, id := range ids {
		if id.Valid() && !slices.Contains(result, id) {
			result = append(result, id)
		}
	}
	return result
}

func elapsedNanoseconds(start time.Time) uint64 {
	return max(uint64(time.Since(start).Nanoseconds()), uint64(time.Nanosecond))
}
