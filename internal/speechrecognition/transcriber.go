package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/hfrepo"
	"overgo/internal/media"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechactivity"
	"overgo/internal/strictjson"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
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

// transcriptionWorkspace owns all reusable mutable storage for one caller.
// Its zero value is ready to use and must not be shared concurrently.
type transcriptionWorkspace struct {
	Frontend   audiodsp.Workspace
	Encoder    Workspace
	Transducer TransducerWorkspace
	frameIDs   []int
	tokens     []int
}

// transcriptionModel owns immutable, validated CPU execution state. It is safe to
// share when each concurrent caller supplies a distinct workspace.
type transcriptionModel struct {
	repository artifact.Repository
	model      artifact.ID
	recipe     recipe.Definition
	contract   modelrecipe.AudioContractDocument
	profile    ExecutionProfile
	frontend   *audiodsp.Frontend
	encoder    *Encoder
	transducer *Transducer
	stream     *audiodsp.StreamFrontend
	memory     uint64
	tokenizer  *hfbpe.Tokenizer
}

func loadTranscriber(ctx context.Context, repository artifact.Repository, definition recipe.Definition, memoryBytes uint64) (*transcriptionModel, error) {
	if definition.Task != recipe.TaskTranscription {
		return nil, errors.New("speech recognition: recipe is not transcription")
	}
	modelID, contractID, profileID, tokenizerID, inventoryID, err := transcriptionDependencies(definition)
	if err != nil {
		return nil, err
	}
	expectedDefinition, err := modelrecipe.TranscriptionDefinition(modelID, contractID, profileID, tokenizerID, inventoryID)
	baseDefinition := expectedDefinition
	checkpointID, adapted := definition.PrimaryDependency(recipe.DependencyCheckpoint)
	if err == nil && adapted {
		expectedDefinition, err = modelrecipe.AdaptedTranscriptionDefinition(baseDefinition, checkpointID)
	}
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
	var encoder *Encoder
	var transducer *Transducer
	var stream *audiodsp.StreamFrontend
	if profile.Transducer == nil {
		encoder, err = LoadEncoder(ctx, hf.Tensors, profile.Encoder, memoryBytes)
		if err == nil && (encoder.InputWidth() != width || profile.BlankToken >= encoder.VocabularySize()) {
			err = errors.New("speech recognition: processor geometry differs from encoder")
		}
	} else {
		if adapted {
			return nil, errors.New("speech recognition: CTC projection checkpoint cannot adapt a recurrent decoder")
		}
		transducer, err = LoadTransducer(ctx, hf.Tensors, profile.Encoder, *profile.Transducer, memoryBytes)
		if err == nil {
			encoder = transducer.encoder
			stream, err = audiodsp.NewStreamFrontend(profile.Frontend, memoryBytes)
		}
	}
	if err != nil {
		return nil, err
	}
	if adapted {
		encoder, err = loadCheckpointProjection(ctx, repository, checkpointID, baseDefinition, profileID, encoder)
		if err != nil {
			return nil, err
		}
	}
	tokenizer, err := hfbpe.Load(modelPath)
	if err != nil {
		return nil, err
	}
	return &transcriptionModel{
		repository: repository, model: modelID, recipe: definition, contract: contract, profile: profile,
		frontend: frontend, encoder: encoder, transducer: transducer, stream: stream, memory: memoryBytes, tokenizer: tokenizer,
	}, nil
}

func loadCheckpointProjection(ctx context.Context, repository artifact.Repository, id artifact.ID, base recipe.Definition, profile artifact.ID, encoder *Encoder) (*Encoder, error) {
	content, err := artifact.RequireTypedContent(ctx, repository, id)
	if err != nil {
		return nil, err
	}
	checkpoint, err := trainingprogram.ParseCheckpoint(content.Data)
	parameters, ok := checked.MulInt(encoder.output.in, encoder.output.in)
	if err != nil || checkpoint.ID() != id || checkpoint.Model != base.Model || !ok || checkpoint.ParameterCount != parameters ||
		!slices.Contains(checkpoint.Processors, profile) ||
		!slices.Contains(checkpoint.Lineage, trainingprogram.LineageParent{Artifact: base.ID, Relation: artifact.RelationDependsOn}) {
		return nil, errors.Join(errors.New("speech recognition: adapter checkpoint authority differs"), err)
	}
	var transformID artifact.ID
	for _, processor := range checkpoint.Processors {
		content, err := artifact.RequireTypedContent(ctx, repository, processor)
		if err != nil {
			return nil, err
		}
		// Training may bind source admission and other processing facts. Serving
		// requires exactly one target transform, not a fixed processor count.
		if content.Descriptor.Schema == trainingdata.TextTransformSchema {
			if transformID.Valid() {
				return nil, errors.New("speech recognition: ambiguous target transformation")
			}
			transformID = processor
		}
	}
	if _, err := trainingdata.RequireTextTransform(ctx, repository, transformID); err != nil {
		return nil, err
	}
	parents, err := repository.Parents(ctx, id)
	if err != nil {
		return nil, err
	}
	expectedParents := append(checkpoint.ArtifactLineage(), artifact.Lineage{Child: id, Parent: checkpoint.Weights, Relation: artifact.RelationContains})
	for _, edge := range expectedParents {
		if !slices.Contains(parents, edge) {
			return nil, errors.New("speech recognition: checkpoint lineage is incomplete")
		}
	}
	directory, err := artifact.AvailablePath(ctx, repository, id, artifact.LocationDirectory)
	if err != nil {
		return nil, err
	}
	loaded, err := trainingprogram.LoadCheckpoint(directory)
	if err != nil || loaded.ID() != id {
		return nil, errors.Join(errors.New("speech recognition: checkpoint publication differs"), err)
	}
	return encoder.WithOutputProjection(ctx, directory, adaptertrain.InputProjectionBinding{
		BaseRecipe: base.ID, TargetTransform: transformID,
	}, checkpoint.Weights)
}

func (transcriber *transcriptionModel) transcribe(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, workspace *transcriptionWorkspace, binding RunBinding, detector *speechactivity.Detector, activityWorkspace *speechactivity.DetectionWorkspace) (recipecontract.Transcription, runrecord.Run, error) {
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
	for _, dependency := range transcriber.recipe.Dependencies {
		if dependency.Role == recipe.DependencyModel {
			inputs = uniqueIDs(append(inputs, dependency.Artifact)...)
		}
	}
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
	phases := completedTranscriptionPhases(decodeDuration, 0, 0, 0)
	spans := []recipecontract.SampleSpan{{End: uint64(len(inspection.Samples))}}
	var activityID artifact.ID
	if detector != nil {
		activityStart := time.Now()
		activity, id, detectErr := detector.DetectDecoded(ctx, inspection.Signal.Source,
			media.DecodedAudio{Format: inspection.Signal.Format, Samples: inspection.Samples}, activityWorkspace)
		phases[1].DurationNS += elapsedNanoseconds(activityStart)
		if detectErr != nil {
			return transcriber.failedExecution(ctx, binding, inputs, "transcription-activity-failed", started, phases, detectErr)
		}
		activityID = id
		inputs = uniqueIDs(append(inputs, activityID)...)
		spans = make([]recipecontract.SampleSpan, len(activity.Segments))
		for i, segment := range activity.Segments {
			spans[i] = segment.Span
		}
		if len(spans) == 0 {
			return transcriber.failedExecution(ctx, binding, inputs, AudioAdmissionFailure, started, phases, ErrAudioAdmissionRefused)
		}
	}
	pieces := make([]transcriptionPiece, 0, len(spans))
	var assembled strings.Builder
	for _, span := range spans {
		if span.Start >= span.End || span.End > uint64(len(inspection.Samples)) {
			return transcriber.failedExecution(ctx, binding, inputs, "transcription-span-invalid", started, phases, errors.New("speech recognition: segment outside admitted samples"))
		}
		samples := inspection.Samples[span.Start:span.End]
		text, measured, failure, decodeErr := transcriber.decodeText(ctx, samples, int(inspection.Signal.Format.SampleRate), workspace)
		for i, metric := range measured {
			phases[i].DurationNS += metric.DurationNS
		}
		if decodeErr != nil {
			return transcriber.failedExecution(ctx, binding, inputs, failure, started, phases, decodeErr)
		}
		if detector == nil {
			assembled.WriteString(text)
		} else {
			text = strings.TrimSpace(text)
			pieces = append(pieces, transcriptionPiece{Span: span, SamplesSHA256: media.SamplesSHA256(samples), Text: text})
			if text != "" {
				if assembled.Len() != 0 {
					assembled.WriteByte(' ')
				}
				assembled.WriteString(text)
			}
		}
	}
	postStart := time.Now()
	result := recipecontract.Transcription{Source: inspection.Signal.Source, Text: assembled.String(), Language: transcriber.profile.Language}
	if err = result.Validate(); err != nil {
		return transcriber.failedExecution(ctx, binding, inputs, "transcription-postprocess-failed", started, phases, err)
	}
	output, err := artifact.JSONContent(transcriptionContract, result)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	contents := []artifact.Content{output}
	outputs := []artifact.ID{output.Descriptor.ID}
	var lineage []artifact.Lineage
	if detector != nil {
		segments, err := artifact.JSONContent(segmentedTranscriptionContract, segmentedTranscription{
			Source: result.Source, Activity: activityID, Pieces: pieces,
		})
		if err != nil {
			return recipecontract.Transcription{}, runrecord.Run{}, err
		}
		contents = append(contents, segments)
		inputs = uniqueIDs(append(inputs, segments.Descriptor.ID)...)
		lineage = artifact.DependencyLineage(segments.Descriptor.ID, transcriber.recipe.ID, activityID, result.Source.Audio, result.Source.Profile)
		lineage = append(lineage, artifact.DependencyLineage(output.Descriptor.ID, segments.Descriptor.ID)...)
	}
	phases[3].DurationNS += elapsedNanoseconds(postStart)
	run, err := runrecord.NewBoundRun(transcriber.recipe.ID, runrecord.OutcomeSucceeded, inputs,
		outputs, "", binding.CodeCommit, binding.Environment,
		elapsedNanoseconds(started), phases)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	batch, err := artifact.NewDocumentBatch(binding.Key, append(contents, runContent), append(lineage, run.Lineage()...), nil)
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

func (transcriber *transcriptionModel) persistRun(ctx context.Context, binding RunBinding, outcome runrecord.Outcome, inputs, outputs []artifact.ID, failure string, measured uint64, phases []runrecord.PhaseMetric) (runrecord.Run, error) {
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

func (transcriber *transcriptionModel) failedExecution(ctx context.Context, binding RunBinding, inputs []artifact.ID, failure string, started time.Time, phases []runrecord.PhaseMetric, executionErr error) (recipecontract.Transcription, runrecord.Run, error) {
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
