package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechsynth"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// SpeechTrainingSpec supplies the exact serving base and normalized target processing contract.
// The training recipe owns dataset membership and the optimizer policy.
type SpeechTrainingSpec struct {
	BaseRecipe  artifact.ID                      `json:"base_recipe"`
	Profile     trainingdata.SpeechLatentProfile `json:"profile"`
	MemoryBytes uint64                           `json:"memory_bytes"`
	Seed        uint64                           `json:"seed"`
	Shuffle     bool                             `json:"shuffle"`
}

func executeSpeechTraining(ctx context.Context, request Request) (result Result, returnErr error) {
	repository, ok := request.Repository.(artifact.Repository)
	if ctx == nil || !ok || repository == nil || request.Speech == nil || request.Audio != nil || request.Recipe.Kind() != artifact.KindRecipe ||
		request.OutputDirectory == "" || request.Steps < 0 || request.ModelDirectory != "" || request.DatasetPath != "" ||
		request.ReferenceDirectory != "" || request.ObjectiveScale != 0 || request.PreferenceMode != "" || request.PreferenceMargin != 0 ||
		request.FreezeLexical || request.MaximumSequence != 0 || request.Rewards != nil || request.Proof {
		return result, errors.New("speech training: exact recipe, speech manifest and new output required; unrelated training inputs refuse")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	spec := *request.Speech
	profile, err := spec.Profile.Content()
	if err != nil || spec.MemoryBytes == 0 || uint64(spec.Profile.MaximumEncodedBytes) > spec.MemoryBytes {
		return result, errors.Join(errors.New("speech training: bounded latent profile required"), err)
	}
	if _, err := os.Stat(request.OutputDirectory); !errors.Is(err, os.ErrNotExist) {
		return result, errors.Join(errors.New("speech training: output must be a new checkpoint directory"), err)
	}
	base, err := recipe.RequireDefinition(ctx, repository, spec.BaseRecipe)
	if err != nil {
		return result, err
	}
	expectedBase, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, base.Model)
	if err != nil || base.ID != expectedBase.ID || spec.Profile.Model != base.Model {
		return result, errors.New("speech training: canonical speech base and target model differ")
	}
	authority, err := ResolveSessionAuthority(ctx, repository, base.Model, request.Recipe)
	if err != nil {
		return result, err
	}
	if authority.Objective != trainingprogram.ObjectiveLatentSequence {
		return result, errors.New("speech training: latent sequence objective required")
	}
	policies, err := trainingprogram.PoliciesFromRecipe(authority.Program.Definition())
	if err != nil {
		return result, err
	}
	objective, err := trainingprogram.LoadObjective(ctx, repository, policies.Objective)
	if err != nil {
		return result, err
	}
	if !slices.Equal(objective.Processors, []artifact.ID{profile.Descriptor.ID}) || len(objective.Projectors) != 0 || len(objective.Codecs) != 0 ||
		!slices.Equal(objective.Signature.Inputs, []recipecontract.Modality{recipecontract.ModalityText}) ||
		!slices.Equal(objective.Signature.Outputs, []recipecontract.Modality{recipecontract.ModalityAudio}) {
		return result, errors.New("speech training: objective must bind the exact text/latent profile")
	}
	processor, err := trainingdata.SpeechLatentProcessor(repository, spec.Profile)
	if err != nil {
		return result, err
	}
	data, err := trainingdata.Materialize(ctx, repository, trainingdata.Authority{Dataset: objective.Dataset, Split: objective.Split,
		Processors: objective.Processors, Signature: objective.Signature, Seed: spec.Seed, Shuffle: spec.Shuffle},
		[]trainingdata.ProcessorBinding{{Artifact: profile.Descriptor.ID, Modalities: []recipecontract.Modality{recipecontract.ModalityText, recipecontract.ModalityAudio}, Process: processor}})
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, data.Close()) }()
	if data.Records() == 0 {
		return result, errors.New("speech training: selected membership is empty")
	}
	var observer *Observer
	if request.Observations != nil {
		observer, err = NewObserver(request.Observations, request.Host)
		if err != nil {
			return result, err
		}
		if err := observer.Admit(ctx, base.Model, request.Recipe); err != nil {
			return result, err
		}
		defer func() {
			id, _, _, err := observer.finishTraining(context.WithoutCancel(ctx), base.Model, request.Recipe, returnErr, result)
			result.Observation, returnErr = id, errors.Join(returnErr, err)
		}()
	}
	result.Objective = trainingprogram.ObjectiveLatentSequence
	result.Backend = "platform-muon"
	if request.Host {
		result.Backend = "host-reference"
	}
	loaded := time.Now()
	directory, err := artifact.AvailablePath(ctx, repository, base.Model, artifact.LocationDirectory)
	if err != nil {
		return result, err
	}
	inventory, err := speechsynth.Inventory(directory)
	if err != nil || inventory.Manifest.ID != base.Model {
		return result, errors.Join(errors.New("speech training: base inventory differs"), err)
	}
	model, err := speechsynth.Load(directory)
	if err != nil {
		return result, err
	}
	facts, err := speechTrainingFacts(model)
	if err != nil {
		return result, err
	}
	if spec.Profile.Width != model.Dims.LatentDim || spec.Profile.Transform != facts[0].Descriptor.ID || spec.Profile.Noise != facts[1].Descriptor.ID {
		return result, errors.New("speech training: normalized target transform, noise or width differs from trainer")
	}
	if err := validateSpeechProducer(ctx, repository, inventory, spec.Profile); err != nil {
		return result, err
	}
	tokenizer, err := speechsynth.LoadTokenizer(filepath.Join(directory, "tokenizer.model"))
	if err != nil {
		return result, err
	}
	_, _, parameters := model.DerivedJointLR(0)
	config, err := authority.Optimizer.Config(parameters)
	if err != nil {
		return result, err
	}
	updates := request.Steps
	if updates == 0 {
		updates = data.Records()
	}
	var resumed trainingprogram.Checkpoint
	var cursor *trainingdata.StreamState
	if request.ResumeDirectory != "" {
		resumed, err = trainingprogram.LoadCheckpoint(request.ResumeDirectory)
		if err != nil {
			return result, err
		}
		cursor = &trainingdata.StreamState{Identity: data.Identity(), Position: resumed.Stream.Position}
	}
	trainer, err := speechsynth.NewJointTrainer(model, config, request.Host)
	if err != nil {
		return result, err
	}
	defer func() { returnErr = errors.Join(returnErr, trainer.Close()) }()
	runSpec := trainingprogram.RunSpec{Recipe: request.Recipe, Initial: trainingprogram.InitialStateSpec{Model: base.Model},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature, Processors: objective.Processors, Policies: policies, Program: trainer.Program()}
	if cursor != nil {
		runSpec.Initial = trainingprogram.InitialStateSpec{Checkpoint: resumed.ID()}
	}
	runPlan, err := trainingprogram.CompileTrainingRunPlanFromRepository(ctx, repository, runSpec)
	if err != nil {
		return result, err
	}
	rng := func(position uint64) []trainingprogram.RNGState {
		return []trainingprogram.RNGState{{Name: "augmentation", Algorithm: spec.Profile.Noise, Counter: position},
			{Name: "data", Algorithm: data.Identity(), Seed: spec.Seed, Counter: position}}
	}
	if cursor != nil {
		if err := trainingprogram.ValidateResume(runPlan, resumed, trainingprogram.ResumeAuthority{Model: base.Model,
			Stream: trainingprogram.DatasetState{Identity: data.Identity(), Position: cursor.Position}, OptimizerPlan: trainer.OptimizerPlan().Identity()}); err != nil {
			return result, err
		}
		if !slices.Contains(resumed.Lineage, trainingprogram.LineageParent{Artifact: base.ID, Relation: artifact.RelationDependsOn}) ||
			!slices.Equal(resumed.RNG, rng(cursor.Position)) {
			return result, errors.New("speech training: base recipe or noise state differs")
		}
		if err := trainer.RestoreCheckpoint(request.ResumeDirectory, resumed.Optimizer); err != nil {
			return result, err
		}
	}
	stream, err := trainingdata.NewStream(data, cursor)
	if err != nil {
		return result, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, DecodeWorkers: 1, MaxBytes: spec.MemoryBytes})
	if err != nil {
		return result, err
	}
	state, err := trainer.OptimizerSnapshot()
	if err != nil {
		return result, err
	}
	result.Optimizer = state.Config
	result.Plan, err = trainingprogram.CompileTrainingSessionPlan(trainingprogram.SessionSpec{Objective: result.Objective, DatasetUnits: data.Records(),
		RequestedUpdates: updates, MaxProjectedWall: request.MaxProjectedWall, Optimizer: state.Config})
	if err != nil {
		return result, err
	}
	observer.Phase(runrecord.PhaseLoad, time.Since(loaded))
	trained := time.Now()
	for step := range updates {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		started := time.Now()
		batch, err := batcher.Next(ctx)
		if err != nil {
			return result, err
		}
		if len(batch.Examples) != 1 {
			return result, errors.New("speech training: expected one sequence per update")
		}
		example := batch.Examples[0]
		text, latents, frames, seed, err := trainingdata.SpeechLatents(example)
		if err != nil {
			return result, err
		}
		tokens, err := tokenizer.Encode(text)
		if err != nil {
			return result, err
		}
		loss, err := trainer.Step(speechsynth.TrainingExample{TextIDs: tokens, Latents: latents, Frames: frames, Seed: seed})
		if err != nil {
			return result, err
		}
		result.Losses = append(result.Losses, loss)
		result.StreamPosition = stream.Snapshot().Position
		observer.SampleStep()
		wall := time.Since(started)
		if request.Progress != nil {
			if _, err := fmt.Fprintf(request.Progress, "step %d/%d source=%s stream_position=%d loss=%g wall=%s backend=%s\n",
				step+1, updates, example.ID, result.StreamPosition, loss, wall, result.Backend); err != nil {
				return result, err
			}
		}
		if err := result.Plan.AdmitStepWall(wall); err != nil {
			return result, err
		}
	}
	observer.Phase(runrecord.PhaseForwardBackward, time.Since(trained))
	if err := ctx.Err(); err != nil {
		return result, err
	}
	state, err = trainer.OptimizerSnapshot()
	if err != nil {
		return result, err
	}
	publication := time.Now()
	var lineage []trainingprogram.LineageParent
	for _, id := range []artifact.ID{base.ID, base.Model, request.Recipe, objective.Dataset, objective.Split,
		profile.Descriptor.ID, spec.Profile.Tokenizer, spec.Profile.Codec, spec.Profile.Transform, spec.Profile.Noise, spec.Profile.Conditioning} {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: id, Relation: artifact.RelationDependsOn})
	}
	result.Checkpoint, err = trainingprogram.PublishCheckpoint(request.OutputDirectory, trainingprogram.CheckpointSpec{
		RunPlan: runPlan.ID(), Program: trainer.Program().ID(), Model: base.Model, Dataset: objective.Dataset, Split: objective.Split,
		Stream: trainingprogram.DatasetState{Identity: data.Identity(), Position: result.StreamPosition}, Optimizer: state, ParameterCount: state.Parameters(),
		RNG: rng(result.StreamPosition), Processors: objective.Processors, Lineage: lineage},
		func(stage string) error {
			return trainer.SaveWeights(filepath.Join(stage, trainingprogram.CheckpointWeights))
		})
	if err != nil {
		return result, err
	}
	batch, err := result.Checkpoint.Batch("training/evidence/speech/"+result.Checkpoint.ID().String(), request.OutputDirectory)
	if err != nil {
		return result, err
	}
	if err := PublishTrainingEvidence(ctx, repository, TrainingEvidencePublication{Batch: batch}); err != nil {
		return result, err
	}
	candidate, err := modelrecipe.AdaptedSpeechDefinition(base, result.Checkpoint.ID())
	if err != nil {
		return result, err
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, repository, "speech/candidate/"+candidate.ID.String(), candidate); err != nil {
		return result, err
	}
	result.Candidate = candidate.ID
	observer.Phase(runrecord.PhasePostprocess, time.Since(publication))
	return result, nil
}

func speechTrainingFacts(model *speechsynth.Model) ([]artifact.Content, error) {
	transform, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/speech-latent-transform/v1"), struct {
		Operation string    `json:"operation"`
		Mean      []float32 `json:"mean"`
		Std       []float32 `json:"std"`
	}{"(codec-mean)/std", model.EmbMean, model.EmbStd})
	if err != nil {
		return nil, err
	}
	noise, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/speech-training-noise/v1"), struct {
		Generator    string `json:"generator"`
		Conditioning string `json:"conditioning"`
	}{"float32(math/rand.New(math/rand.NewSource(record.seed)).NormFloat64()) in frame/channel order", "text embeddings; BOS; prior normalized targets"})
	return []artifact.Content{transform, noise}, err
}

func validateSpeechProducer(ctx context.Context, reader artifact.Reader, inventory modelartifact.Inventory, profile trainingdata.SpeechLatentProfile) error {
	var tokenizer, codec artifact.ID
	for _, component := range inventory.Manifest.Components {
		switch component.Role {
		case artifact.ComponentTokenizer:
			tokenizer = component.Artifact
		case artifact.ComponentWeights:
			codec = component.Artifact
		}
	}
	if profile.Tokenizer != tokenizer || profile.Codec != codec {
		return errors.New("speech training: tokenizer or codec weight identity differs from base inventory")
	}
	for _, id := range []artifact.ID{profile.Transform, profile.Noise, profile.Conditioning} {
		if _, found, err := reader.Artifact(ctx, id); err != nil || !found {
			return errors.Join(fmt.Errorf("speech training: target provenance %s is absent", id), err)
		}
	}
	return nil
}
