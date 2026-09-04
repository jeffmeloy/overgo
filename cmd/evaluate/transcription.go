package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

// transcriptionManifest deliberately keeps target-bearing suite data and
// target-free prediction data in separate files. The manifest binds both to
// the exact execution authorities expected on every stored prediction run.
type transcriptionManifest struct {
	Suite           string      `json:"suite"`
	Predictions     string      `json:"predictions"`
	ModelDefinition artifact.ID `json:"model_definition"`
	RuntimeRecipe   artifact.ID `json:"runtime_recipe"`
	CodeCommit      string      `json:"code_commit"`
	Environment     artifact.ID `json:"environment"`
}

type transcriptionResourceManifest struct {
	Suite           string                                  `json:"suite"`
	Model           artifact.ID                             `json:"model"`
	ModelDefinition artifact.ID                             `json:"model_definition"`
	RuntimeRecipe   artifact.ID                             `json:"runtime_recipe"`
	CodeCommit      string                                  `json:"code_commit"`
	Environment     artifact.ID                             `json:"environment"`
	MemoryBytes     uint64                                  `json:"memory_bytes"`
	Options         evaluation.TranscriptionResourceOptions `json:"options"`
	Inputs          []transcriptionResourceManifestInput    `json:"inputs"`
}

type transcriptionResourceManifestInput struct {
	Name   string                        `json:"name"`
	Path   string                        `json:"path"`
	Origin dataset.AudioPayloadOrigin    `json:"origin"`
	Policy dataset.AudioInspectionPolicy `json:"policy"`
}

func evaluateTranscriptionManifest(ctx context.Context, repository, path string) error {
	if ctx == nil {
		return errors.New("evaluate: transcription context is absent")
	}
	manifestData, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var manifest transcriptionManifest
	if err = strictjson.DecodeBytes(manifestData, &manifest); err != nil {
		return fmt.Errorf("evaluate: decode transcription manifest: %w", err)
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return err
	}
	manifest.Suite = resolveEvaluationPath(base, manifest.Suite)
	manifest.Predictions = resolveEvaluationPath(base, manifest.Predictions)
	if manifest.Suite == "" || manifest.Predictions == "" {
		return errors.New("evaluate: transcription suite or predictions are absent")
	}
	suiteData, err := os.ReadFile(manifest.Suite)
	if err != nil {
		return err
	}
	var suite evaluation.TranscriptionSuite
	if err = strictjson.DecodeBytes(suiteData, &suite); err != nil {
		return fmt.Errorf("evaluate: decode transcription suite: %w", err)
	}
	predictionData, err := os.ReadFile(manifest.Predictions)
	if err != nil {
		return err
	}
	var predictions []evaluation.TranscriptionPrediction
	if err = strictjson.DecodeBytes(predictionData, &predictions); err != nil {
		return fmt.Errorf("evaluate: decode transcription predictions: %w", err)
	}
	compiled, err := evaluation.CompileTranscription(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{
		ModelDefinition: manifest.ModelDefinition, RuntimeRecipe: manifest.RuntimeRecipe,
		CodeCommit: strings.TrimSpace(manifest.CodeCommit), Environment: manifest.Environment,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := evaluation.EvaluateTranscription(ctx, store, compiled, plan, predictions)
	if err != nil {
		return err
	}
	fmt.Printf("transcription report %s plan=%s utterances=%d wer=%.6f cer=%.6f admission_failures=%d inference_failures=%d silent_control_failures=%d\n",
		report.ID, report.Plan, report.Overall.Utterances, report.Overall.WordErrorRate, report.Overall.CharacterErrorRate,
		report.Overall.AdmissionFailures, report.Overall.InferenceFailures, report.Overall.SilentControlFailures,
	)
	return nil
}

func evaluateTranscriptionResourceManifest(ctx context.Context, repository, path string) error {
	if ctx == nil {
		return errors.New("evaluate: transcription resource context is absent")
	}
	manifestData, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var manifest transcriptionResourceManifest
	if err = strictjson.DecodeBytes(manifestData, &manifest); err != nil {
		return fmt.Errorf("evaluate: decode transcription resource manifest: %w", err)
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return err
	}
	suiteData, err := os.ReadFile(resolveEvaluationPath(base, manifest.Suite))
	if err != nil {
		return err
	}
	var suite evaluation.TranscriptionSuite
	if err = strictjson.DecodeBytes(suiteData, &suite); err != nil {
		return fmt.Errorf("evaluate: decode transcription resource suite: %w", err)
	}
	inputs := make([]evaluation.TranscriptionResourceInput, len(manifest.Inputs))
	for index, input := range manifest.Inputs {
		data, readErr := os.ReadFile(resolveEvaluationPath(base, input.Path))
		if readErr != nil {
			return readErr
		}
		inputs[index] = evaluation.TranscriptionResourceInput{
			Name: input.Name, Data: data, Origin: input.Origin, Policy: input.Policy,
		}
	}
	compiled, err := evaluation.CompileTranscription(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{
		ModelDefinition: manifest.ModelDefinition, RuntimeRecipe: manifest.RuntimeRecipe,
		CodeCommit: strings.TrimSpace(manifest.CodeCommit), Environment: manifest.Environment,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := evaluation.EvaluateTranscriptionResources(
		ctx, store, compiled, plan, manifest.Model, inputs, manifest.Options,
		func(loadContext context.Context) (evaluation.TranscriptionResourceExecutor, error) {
			return speechrecognition.LoadTranscriber(loadContext, store, manifest.RuntimeRecipe, manifest.MemoryBytes)
		},
	)
	if err != nil {
		return err
	}
	fmt.Printf("transcription resource report %s plan=%s quality=%s runs=%d audio_seconds=%.6f wall_seconds=%.6f rtf=%.6f allocations=%d peak_host_bytes=%d admission_failures=%d inference_failures=%d\n",
		report.ID, report.Plan, report.Quality, report.Summary.Runs, report.Summary.AudioSeconds,
		report.Summary.WallSeconds, report.Summary.RealTimeFactor, report.Summary.Allocations,
		report.Summary.PeakHostBytes, report.Summary.AdmissionFailures, report.Summary.InferenceFailures,
	)
	return nil
}

func resolveEvaluationPath(base, value string) string {
	value = strings.TrimSpace(value)
	if value == "" || filepath.IsAbs(value) {
		return value
	}
	return filepath.Join(base, value)
}
