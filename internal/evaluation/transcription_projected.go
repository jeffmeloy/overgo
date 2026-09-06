package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/speechrecognition"
	"overgo/internal/tokenizer"
)

type projectedTranscriptionRuntime struct {
	repository            artifact.Repository
	runner                *inference.Runner
	projection            projector.Session
	definition            recipe.Definition
	decodeRecipe          artifact.ID
	plan                  artifact.ID
	model, projector      artifact.ID
	prompt                string
	maxTokens, sampleRate int
}

func loadProjectedTranscription(ctx context.Context, repository artifact.Repository, plan Plan, prompt string, maxTokens int, decodeRecipe artifact.ID) (transcriptionRuntime, error) {
	definition, err := recipe.RequireDefinition(ctx, repository, plan.body.RuntimeRecipe)
	if err != nil {
		return nil, err
	}
	if definition.Task != recipe.TaskProjection {
		return nil, errors.New("evaluation: projected transcription requires an exact projection recipe")
	}
	modelID, hasModel := definition.PrimaryDependency(recipe.DependencyModel)
	projectorID, hasProjector := definition.PrimaryDependency(recipe.DependencyProjector)
	if !hasModel || !hasProjector {
		return nil, errors.New("evaluation: projection recipe dependencies are incomplete")
	}
	modelPath, err := transcriptionGGUFPath(ctx, repository, modelID)
	if err != nil {
		return nil, err
	}
	projectorPath, err := transcriptionGGUFPath(ctx, repository, projectorID)
	if err != nil {
		return nil, err
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, repository, modelPath)
	if err != nil {
		return nil, err
	}
	identity, err := loaded.Identity()
	if err != nil || identity.Model != modelID || identity.Definition != plan.body.ModelDefinition || identity.Recipe != decodeRecipe {
		return nil, errors.Join(errors.New("evaluation: active language model differs from transcription plan"), err, loaded.Close())
	}
	inventory, media, processor, err := projector.InspectProjection(ctx, projectorPath)
	if err != nil {
		return nil, errors.Join(err, loaded.Close())
	}
	boundProcessor, hasProcessor := definition.PrimaryDependency(recipe.DependencyProcessorProfile)
	processor, err = projector.ResolvePreprocessProfile(ctx, repository, definition, processor)
	if err != nil {
		return nil, errors.Join(err, loaded.Close())
	}
	if inventory.Manifest.ID != projectorID || hasProcessor != (processor != nil) || processor != nil && processor.ID != boundProcessor {
		return nil, errors.Join(errors.New("evaluation: projector or preprocessing differs from recipe"), loaded.Close())
	}
	expected, err := modelrecipe.ProjectionDefinition(modelID, projectorID, boundProcessor, media...)
	if err != nil || expected.ID != definition.ID {
		return nil, errors.Join(errors.New("evaluation: projection execution graph differs from declared artifact semantics"), err, loaded.Close())
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return nil, errors.Join(err, loaded.Close())
	}
	projection, err := projector.OpenSession(ctx, projectorPath, projector.OpenOptions{CUDA: true, MediaPreprocess: processor})
	if err != nil {
		return nil, errors.Join(err, runner.Close())
	}
	runtime := &projectedTranscriptionRuntime{repository: repository, runner: runner, projection: projection, definition: definition,
		decodeRecipe: identity.Recipe, plan: plan.identity, model: modelID, projector: projectorID, prompt: prompt, maxTokens: maxTokens}
	runtime.sampleRate, err = projector.AudioSampleRate(projection)
	if err != nil || !projection.Capabilities().Audio {
		return nil, errors.Join(errors.New("evaluation: projection does not admit waveform input"), err, runtime.close())
	}
	return runtime, nil
}

// Follow the registered logical artifact to its first GGUF component. GGUF's
// existing loader validates and joins any remaining shards.
func transcriptionGGUFPath(ctx context.Context, reader artifact.Reader, id artifact.ID) (string, error) {
	manifest, found, err := reader.Manifest(ctx, id)
	if err != nil {
		return "", err
	}
	if !found {
		return "", errors.New("evaluation: transcription model manifest is absent")
	}
	for _, component := range manifest.Components {
		if component.Role == artifact.ComponentWeights || component.Role == artifact.ComponentWeightsShard && component.Ordinal == 0 {
			return artifact.AvailablePath(ctx, reader, component.Artifact, artifact.LocationFile)
		}
	}
	return "", errors.New("evaluation: transcription model has no registered GGUF weights")
}

func (runtime *projectedTranscriptionRuntime) close() error {
	return errors.Join(runtime.projection.Close(), runtime.runner.Close())
}

func (runtime *projectedTranscriptionRuntime) transcribe(ctx context.Context, input TranscriptionResourceInput, data []byte, binding speechrecognition.RunBinding) (recipecontract.Transcription, runrecord.Run, error) {
	started := time.Now()
	inspection, err := dataset.InspectAudio(ctx, runtime.repository, data, input.Reference.Origin, input.Policy)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	phases := []runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: elapsedResourceNanoseconds(started)}}
	inputs := []artifact.ID{runtime.plan, runtime.model, runtime.projector, runtime.decodeRecipe, runtime.runner.RuntimePolicy().ID, binding.Dataset, binding.Split,
		inspection.Signal.Source.Audio, inspection.Signal.Source.Profile, inspection.SignalID, inspection.PolicyID, inspection.DecisionID}
	policyContent, err := runtime.runner.RuntimePolicy().Content()
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	finish := func(result recipecontract.Transcription, failure string, cause error) (recipecontract.Transcription, runrecord.Run, error) {
		outcome := runrecord.OutcomeFailed
		var outputs []artifact.ID
		contents := []artifact.Content{policyContent}
		if cause == nil {
			if err := result.Validate(); err != nil {
				return result, runrecord.Run{}, err
			}
			content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), result)
			if err != nil {
				return result, runrecord.Run{}, err
			}
			contents = append(contents, content)
			outputs = append(outputs, content.Descriptor.ID)
			outcome = runrecord.OutcomeSucceeded
		}
		run, err := runrecord.NewBoundRun(runtime.definition.ID, outcome, inputs, outputs, failure, binding.CodeCommit, binding.Environment, elapsedResourceNanoseconds(started), phases)
		if err != nil {
			return result, runrecord.Run{}, errors.Join(cause, err)
		}
		content, err := run.Content()
		if err != nil {
			return result, runrecord.Run{}, errors.Join(cause, err)
		}
		contents = append(contents, content)
		batch, err := artifact.NewDocumentBatch(binding.Key, contents, run.Lineage(), nil)
		if err == nil {
			_, err = artifact.CommitBatch(ctx, runtime.repository, batch)
		}
		if err != nil && !errors.Is(err, artifact.ErrNoChange) {
			return result, runrecord.Run{}, errors.Join(cause, err)
		}
		return result, run, cause
	}
	if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
		return finish(recipecontract.Transcription{}, speechrecognition.AudioAdmissionFailure, speechrecognition.ErrAudioAdmissionRefused)
	}
	if inspection.Signal.Format.Channels != 1 || inspection.Signal.Format.SampleRate != uint64(runtime.sampleRate) {
		return finish(recipecontract.Transcription{}, speechrecognition.AudioFormatFailure, errors.New("evaluation: waveform differs from projector format"))
	}
	prepareStarted := time.Now()
	prompt, err := runtime.projection.BuildAudioPrompt(ctx, runtime.runner, inspection.Samples, "", runtime.prompt)
	if err != nil {
		return finish(recipecontract.Transcription{}, "transcription-prepare-failed", err)
	}
	ids, projected, err := inference.CompileProjectedInputs(prompt, runtime.runner.Spec().EmbeddingLength)
	if err != nil {
		return finish(recipecontract.Transcription{}, "transcription-prepare-failed", err)
	}
	phases = append(phases, runrecord.PhaseMetric{Phase: runrecord.PhasePrepare, DurationNS: elapsedResourceNanoseconds(prepareStarted)})
	var text strings.Builder
	var tokens []tokenizer.TokenID
	generateStarted := time.Now()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return finish(recipecontract.Transcription{}, "transcription-prepare-failed", err)
	}
	_, _, err = runtime.runner.Generate(ctx, "", inference.GenerateOptions{PromptTokenIDs: ids, ProjectedInputs: &projected, MaxNewTokens: runtime.maxTokens, DeviceGreedy: true,
		Sampler: greedy,
		OnToken: func(event inference.TokenEvent) error {
			tokens = append(tokens, event.ID)
			text.WriteString(event.Piece)
			return nil
		}})
	phases = append(phases, runrecord.PhaseMetric{Phase: runrecord.PhaseGenerate, DurationNS: elapsedResourceNanoseconds(generateStarted)})
	if err != nil {
		return finish(recipecontract.Transcription{}, "transcription-inference-failed", err)
	}
	if err = completeProjectedTranscript(tokens, runtime.runner.SamplingEOGTokens(), runtime.maxTokens); err != nil {
		return finish(recipecontract.Transcription{}, "transcription-output-budget-exhausted", err)
	}
	return finish(recipecontract.Transcription{Source: inspection.Signal.Source, Text: text.String()}, "", nil)
}

func completeProjectedTranscript(tokens, stopIDs []tokenizer.TokenID, maximum int) error {
	if len(tokens) == 0 || len(tokens) > maximum || !slices.Contains(stopIDs, tokens[len(tokens)-1]) {
		return fmt.Errorf("evaluation: transcription did not terminate within its %d-token budget", maximum)
	}
	return nil
}
