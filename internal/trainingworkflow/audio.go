package trainingworkflow

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"slices"
	"time"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// AudioTrainingSpec supplies source processing and the exact frozen base recipe.
// The active training recipe's objective owns dataset, split and processor IDs.
// Lowercase is required even when false; no target convention is inferred from
// a model name. MemoryBytes is the shared per-component numeric admission ceiling,
// not a measured process peak. No augmentation is applied.
type AudioTrainingSpec struct {
	BaseRecipe  artifact.ID                   `json:"base_recipe"`
	MemoryBytes uint64                        `json:"memory_bytes"`
	Inspection  dataset.AudioInspectionPolicy `json:"inspection"`
	Lowercase   *bool                         `json:"lowercase"`
	Seed        uint64                        `json:"seed"`
	Shuffle     bool                          `json:"shuffle"`
}

// The shared CTC adapter applies one complete sequence per optimizer update;
// decoding is serial because its payload reader and lease own mutable scratch.
const audioTrainingExamples = 1

func executeAudioTraining(ctx context.Context, request Request) (result Result, returnErr error) {
	repository, writable := request.Repository.(artifact.Repository)
	if ctx == nil || !writable || repository == nil || request.Audio == nil || request.Recipe.Kind() != artifact.KindRecipe ||
		request.OutputDirectory == "" || request.Steps < 0 || request.ModelDirectory != "" || request.DatasetPath != "" ||
		request.ReferenceDirectory != "" || request.ObjectiveScale != 0 || request.FreezeLexical || request.MaximumSequence != 0 {
		return result, errors.New("audio training: writable repository, exact recipe and output required; dense-only inputs refuse")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	spec := *request.Audio
	if spec.BaseRecipe.Kind() != artifact.KindRecipe || spec.Lowercase == nil || spec.MemoryBytes == 0 ||
		spec.Inspection.MaximumEncodedBytes > spec.MemoryBytes || spec.Inspection.MaximumSamples > spec.MemoryBytes/binaryschema.Uint32Bytes {
		return result, errors.New("audio training: explicit base, target case and bounded signal policy required")
	}
	if err := spec.Inspection.Validate(); err != nil {
		return result, err
	}
	if _, err := os.Stat(request.OutputDirectory); !errors.Is(err, os.ErrNotExist) {
		return result, errors.Join(errors.New("audio training: output must be a new checkpoint directory"), err)
	}
	base, err := recipe.RequireDefinition(ctx, repository, spec.BaseRecipe)
	if err != nil {
		return result, err
	}
	authority, err := ResolveSessionAuthority(ctx, repository, base.Model, request.Recipe)
	if err != nil {
		return result, err
	}
	programObjective, err := ProgramObjective(authority.Program)
	if err != nil || programObjective != trainingprogram.ObjectiveTokenPrediction || authority.Objective != trainingprogram.ObjectiveCTC {
		return result, errors.Join(errors.New("audio training: requires the shared forward/backward/optimize chain and a bound CTC objective"), err)
	}
	policies, err := trainingprogram.PoliciesFromRecipe(authority.Program.Definition())
	if err != nil {
		return result, err
	}
	objective, err := trainingprogram.LoadObjective(ctx, repository, policies.Objective)
	if err != nil {
		return result, err
	}
	if !slices.Equal(objective.Signature.Inputs, []recipecontract.Modality{recipecontract.ModalityAudio}) ||
		!slices.Equal(objective.Signature.Outputs, []recipecontract.Modality{recipecontract.ModalityText}) {
		return result, errors.New("audio training: objective must bind one audio input and one text target")
	}
	transform, err := trainingdata.NewTextTransform(*spec.Lowercase)
	if err != nil {
		return result, err
	}
	transformContent, err := transform.Content()
	if err != nil {
		return result, err
	}
	processorContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/audio-text-processor/v1"), spec.Inspection)
	if err != nil {
		return result, err
	}
	profile, found := base.PrimaryDependency(recipe.DependencyProcessorProfile)
	processors := []artifact.ID{profile, transform.ID, processorContent.Descriptor.ID}
	slices.SortFunc(processors, artifact.CompareID)
	if !found || !slices.Equal(processors, objective.Processors) || len(objective.Projectors) != 0 || len(objective.Codecs) != 0 {
		return result, errors.New("audio training: objective processing authorities differ")
	}
	if err := publishAudioTrainingFacts(ctx, repository, artifact.Batch{Key: "processors/" + request.Recipe.String(), Contents: []artifact.Content{transformContent, processorContent}}); err != nil {
		return result, err
	}
	source, err := dataset.NewAudioPayloadReader(spec.MemoryBytes)
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, source.Close()) }()
	processor, err := trainingdata.AudioTextProcessor(repository, source, spec.Inspection)
	if err != nil {
		return result, err
	}
	dataAuthority := trainingdata.Authority{Dataset: objective.Dataset, Split: objective.Split, Seed: spec.Seed, Shuffle: spec.Shuffle,
		Processors: []artifact.ID{processorContent.Descriptor.ID}, Signature: objective.Signature}
	data, err := trainingdata.Materialize(ctx, repository, dataAuthority, []trainingdata.ProcessorBinding{{Artifact: processorContent.Descriptor.ID,
		Modalities: []recipecontract.Modality{recipecontract.ModalityAudio, recipecontract.ModalityText}, Process: processor}})
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, data.Close()) }()
	if data.Records() == 0 {
		return result, errors.New("audio training: empty selected dataset")
	}
	orderContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/audio-training-order/v1"), struct {
		Seed    uint64 `json:"seed"`
		Shuffle bool   `json:"shuffle"`
		Augment bool   `json:"augment"`
	}{Seed: spec.Seed, Shuffle: spec.Shuffle})
	if err != nil {
		return result, err
	}
	if err := publishAudioTrainingFacts(ctx, repository, artifact.Batch{Key: "order/" + orderContent.Descriptor.ID.String(), Contents: []artifact.Content{orderContent}}); err != nil {
		return result, err
	}
	rng := func(position uint64) []trainingprogram.RNGState {
		return []trainingprogram.RNGState{{Name: "augmentation", Algorithm: orderContent.Descriptor.ID}, {Name: "data", Algorithm: orderContent.Descriptor.ID, Seed: spec.Seed, Counter: position}}
	}
	var observer *Observer
	if request.Observations != nil {
		observer, err = NewObserver(request.Observations, true)
		if err != nil {
			return result, err
		}
		if err := observer.Admit(ctx, base.Model, request.Recipe); err != nil {
			return result, err
		}
		defer func() {
			result.Objective = trainingprogram.ObjectiveCTC
			id, _, _, err := observer.finishTraining(context.WithoutCancel(ctx), base.Model, request.Recipe, returnErr, result)
			result.Observation, returnErr = id, errors.Join(returnErr, err)
		}()
	}
	loadStarted := time.Now()
	session, err := speechrecognition.LoadSession(ctx, repository, base.ID, spec.MemoryBytes)
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, session.Close(context.WithoutCancel(ctx))) }()
	lease, err := session.Lease(ctx)
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, lease.Release()) }()
	adapter, err := lease.NewOutputAdapter(ctx, spec.Inspection.MaximumSamples, authority.Optimizer)
	if err != nil {
		return result, err
	}
	observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	binding := adaptertrain.InputProjectionBinding{BaseRecipe: base.ID, TargetTransform: transform.ID}
	runSpec := trainingprogram.RunSpec{Recipe: request.Recipe, Initial: trainingprogram.InitialStateSpec{Model: base.Model}, Dataset: objective.Dataset, Split: objective.Split,
		Signature: objective.Signature, Processors: processors, Policies: policies, Program: adapter.Program()}
	var resumed trainingprogram.Checkpoint
	var resumeStream *trainingdata.StreamState
	if request.ResumeDirectory != "" {
		resumed, err = trainingprogram.LoadCheckpoint(request.ResumeDirectory)
		if err != nil {
			return result, err
		}
		runSpec.Initial = trainingprogram.InitialStateSpec{Checkpoint: resumed.ID()}
		resumeStream = &trainingdata.StreamState{Identity: data.Identity(), Position: resumed.Stream.Position}
	}
	runPlan, err := trainingprogram.CompileTrainingRunPlanFromRepository(ctx, repository, runSpec)
	if err != nil {
		return result, err
	}
	if resumeStream != nil {
		_, err := RestoreInputProjection(request.ResumeDirectory, adapter, binding, runPlan, trainingprogram.ResumeAuthority{Model: base.Model,
			Stream: trainingprogram.DatasetState{Identity: data.Identity(), Position: resumeStream.Position}, OptimizerPlan: adapter.Program().OptimizerIdentity()}, rng(resumeStream.Position))
		if err != nil {
			return result, err
		}
	}
	stream, err := trainingdata.NewStream(data, resumeStream)
	if err != nil {
		return result, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: audioTrainingExamples, DecodeWorkers: audioTrainingExamples, MaxBytes: spec.MemoryBytes})
	if err != nil {
		return result, err
	}
	state, err := adapter.OptimizerSnapshot()
	if err != nil {
		return result, err
	}
	result.Plan, err = trainingprogram.CompileTrainingSessionPlan(trainingprogram.SessionSpec{Objective: trainingprogram.ObjectiveCTC, DatasetUnits: data.Records(),
		RequestedUpdates: request.Steps, MaxProjectedWall: request.MaxProjectedWall, Optimizer: state.Config})
	if err != nil {
		return result, err
	}
	result.Backend, result.Objective, result.Optimizer = "host-reference", trainingprogram.ObjectiveCTC, state.Config
	request.Steps = result.Plan.Updates()
	guard := stepGuard(request, result.Backend)
	execution, err := adapter.Bind(ctx)
	if err != nil {
		return result, err
	}
	trainStarted := time.Now()
	for step := range request.Steps {
		started := time.Now()
		batch, err := batcher.Next(ctx)
		if err != nil {
			return result, err
		}
		if len(batch.Examples) != audioTrainingExamples {
			return result, errors.New("audio training: source processor returned an invalid batch")
		}
		example := batch.Examples[0]
		if len(example.Values) != len(objective.Signature.Inputs)+len(objective.Signature.Outputs) || example.Values[0].Role != trainingdata.RoleInput || example.Values[1].Role != trainingdata.RoleTarget || example.Values[1].Modality != recipecontract.ModalityText || example.Values[1].Encoding != trainingdata.EncodingUTF8 {
			return result, errors.New("audio training: source processor returned an invalid pair")
		}
		samples, rate, err := trainingdata.Audio(example.Values[0])
		if err != nil {
			return result, err
		}
		raw := string(example.Values[1].Data)
		target, err := transform.Apply(raw)
		if err != nil {
			return result, err
		}
		prepared, err := lease.PrepareCTC(ctx, samples, rate, target)
		if err != nil {
			return result, err
		}
		if err := execution.Run(&prepared); err != nil {
			return result, err
		}
		result.Losses = append(result.Losses, prepared.Loss)
		result.StreamPosition = stream.Snapshot().Position
		observer.SampleStep()
		if err := guard(step, prepared.Loss, time.Since(started)); err != nil {
			return result, err
		}
		if request.Progress != nil {
			if _, err := fmt.Fprintf(request.Progress, "audio source=%s raw_text_sha256=%x stream_position=%d loss=%g\n", example.ID, sha256.Sum256(example.Values[1].Data), result.StreamPosition, prepared.Loss); err != nil {
				return result, err
			}
		}
	}
	observer.Phase(runrecord.PhaseForwardBackward, time.Since(trainStarted))
	if err := ctx.Err(); err != nil {
		return result, err
	}
	streamState := stream.Snapshot()
	result.Checkpoint, err = PublishInputProjection(request.OutputDirectory, adapter, binding, trainingprogram.CheckpointSpec{RunPlan: runPlan.ID(), Program: adapter.Program().ID(),
		Model: base.Model, Dataset: objective.Dataset, Split: objective.Split, Stream: trainingprogram.DatasetState{Identity: streamState.Identity, Position: streamState.Position},
		RNG: rng(streamState.Position), Processors: runPlan.Processors(), Lineage: []trainingprogram.LineageParent{{Artifact: base.Model, Relation: artifact.RelationDependsOn},
			{Artifact: objective.Dataset, Relation: artifact.RelationDependsOn}, {Artifact: objective.Split, Relation: artifact.RelationDependsOn}}})
	if err != nil {
		return result, err
	}
	batch, err := result.Checkpoint.Batch("checkpoint/"+result.Checkpoint.ID().String(), request.OutputDirectory)
	if err != nil {
		return result, err
	}
	if err := publishAudioTrainingFacts(ctx, repository, batch); err != nil {
		return result, err
	}
	candidate, err := modelrecipe.AdaptedTranscriptionDefinition(base, result.Checkpoint.ID())
	if err != nil {
		return result, err
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, repository, "training/audio/candidate/"+candidate.ID.String(), candidate); err != nil {
		return result, err
	}
	result.Candidate = candidate.ID
	return result, nil
}

func publishAudioTrainingFacts(ctx context.Context, repository artifact.Repository, batch artifact.Batch) error {
	batch.Key = "training/evidence/audio/" + batch.Key
	err := PublishTrainingEvidence(ctx, repository, TrainingEvidencePublication{Batch: batch})
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}
