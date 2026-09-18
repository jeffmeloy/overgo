package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/tokenizer"
)

func verifyInference(
	repository, path, input string,
	override modelintake.SessionOverride,
	residency recipe.ResidencyPolicy,
) error {
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	var suite evaluation.ExactSuite
	if err := json.Unmarshal([]byte(input), &suite); err != nil {
		return fmt.Errorf("recipe: decode inference suite: %w", err)
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return fmt.Errorf("recipe: compile inference suite: %w", err)
	}
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := prepareExactCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	runtime := &deferredExactRuntime{open: func(ctx context.Context) (exactRuntime, error) {
		return openExactCandidate(ctx, path, candidate)
	}}
	return verifyExactRuntime(ctx, store, candidate.Definition, revision, suite, exactPlan, candidate.Resolved.Document.ID, runtime)
}

func prepareExactCandidate(ctx context.Context, store *overgodb.Store, path string,
	override modelintake.SessionOverride, residency recipe.ResidencyPolicy,
) (modelintake.Candidate, error) {
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return modelintake.Candidate{}, err
	}
	if err := modelintake.RegisterCandidate(ctx, store, candidate); err != nil {
		return modelintake.Candidate{}, err
	}
	return candidate, nil
}

func openExactCandidate(ctx context.Context, path string, candidate modelintake.Candidate) (*inference.Runner, error) {
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved)
	if err != nil {
		return nil, err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return nil, errors.Join(err, loaded.Close())
	}
	return runner, nil
}

// Projection cases carry exact source identities inside the existing exact suite.
// Audio fixtures are headerless little-endian float32 at the projector's rate.
type projectionFile struct {
	Kind     string      `json:"kind"`
	Path     string      `json:"path"`
	Artifact artifact.ID `json:"artifact"`
}

type projectionInput struct {
	Kind  string           `json:"kind"`
	Files []projectionFile `json:"files"`
	Text  []string         `json:"text"`
	FPS   float64          `json:"fps,omitzero"`
}

type projectedExactRuntime struct {
	runner     *inference.Runner
	projection projector.Session
}

// Close releases the projection and language runtimes, retaining both errors.
func (runtime *projectedExactRuntime) Close() error {
	return errors.Join(runtime.projection.Close(), runtime.runner.Close())
}

func projectionModes(capabilities projector.SessionCapabilities) map[string]bool {
	return map[string]bool{"image": capabilities.Image, "images": capabilities.MultiImage,
		"audio": capabilities.Audio, "video": capabilities.Video, "mixed": capabilities.MediaHistory}
}

func decodeProjectionInput(raw string, capabilities projector.SessionCapabilities) (projectionInput, error) {
	var input projectionInput
	if err := json.Unmarshal([]byte(raw), &input); err != nil {
		return input, err
	}
	if !projectionModes(capabilities)[input.Kind] || len(input.Files) == 0 || strings.TrimSpace(strings.Join(input.Text, "")) == "" {
		return input, errors.New("projection: undeclared mode or empty fixture")
	}
	images, audio := 0, 0
	for _, file := range input.Files {
		if file.Path == "" || file.Artifact.Kind() != artifact.KindFile {
			return input, errors.New("projection: fixture requires a path and exact file identity")
		}
		switch file.Kind {
		case "image":
			images++
		case "audio-f32le":
			audio++
		default:
			return input, errors.New("projection: unsupported fixture encoding")
		}
	}
	valid := len(input.Text) == 2
	switch input.Kind {
	case "image":
		valid = valid && images == 1 && audio == 0
	case "images":
		valid = images >= 2 && audio == 0 && len(input.Text) == images+1
	case "audio":
		valid = valid && audio == 1 && images == 0
	case "video":
		valid = valid && images >= 2 && audio == 0 && input.FPS > 0 && !math.IsInf(input.FPS, 0) && !math.IsNaN(input.FPS)
	case "mixed":
		valid = images > 0 && audio > 0 && len(input.Text) == len(input.Files)+1
	}
	if !valid {
		return input, errors.New("projection: fixture does not cover its declared input mode")
	}
	return input, nil
}

func requireProjectionCoverage(suite evaluation.ExactSuite, capabilities projector.SessionCapabilities) error {
	remaining := projectionModes(capabilities)
	for _, test := range suite.Cases {
		input, err := decodeProjectionInput(test.Prompt, capabilities)
		if err != nil {
			return fmt.Errorf("projection case %s: %w", test.Name, err)
		}
		delete(remaining, input.Kind)
	}
	for mode, required := range remaining {
		if required {
			return fmt.Errorf("projection: declared %s mode has no executed-case contract", mode)
		}
	}
	return nil
}

// Generate validates fixture identities and executes projection-to-language generation.
func (runtime *projectedExactRuntime) Generate(ctx context.Context, raw string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	input, err := decodeProjectionInput(raw, runtime.projection.Capabilities())
	if err != nil {
		return nil, "", err
	}
	var pictures []image.Image
	var waveform []float32
	var ordered []projector.MediaInput
	for _, file := range input.Files {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
		data, err := artifact.ReadContentFile(file.Path)
		if err != nil {
			return nil, "", err
		}
		identity, err := artifact.IdentifyBytes(artifact.KindFile, data)
		if err != nil || identity != file.Artifact {
			return nil, "", errors.New("projection: fixture bytes differ from the exact suite")
		}
		if file.Kind == "image" {
			picture, _, err := image.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, "", err
			}
			pictures = append(pictures, picture)
			ordered = append(ordered, projector.NewImageMediaInput(picture))
		} else {
			width := binary.Size(float32(0))
			if len(data)%width != 0 {
				return nil, "", errors.New("projection: incomplete float32 audio scalar")
			}
			waveform = make([]float32, len(data)/width)
			if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, waveform); err != nil {
				return nil, "", err
			}
			ordered = append(ordered, projector.NewAudioMediaInput(waveform))
		}
	}
	var prompt projector.MultimodalPrompt
	switch input.Kind {
	case "image":
		prompt, err = runtime.projection.BuildImagePrompt(ctx, runtime.runner, pictures[0], input.Text[0], input.Text[1], false)
	case "images":
		prompt, err = runtime.projection.BuildImagesPrompt(ctx, runtime.runner, pictures, input.Text, projector.PromptOptions{})
	case "audio":
		prompt, err = runtime.projection.BuildAudioPrompt(ctx, runtime.runner, waveform, input.Text[0], input.Text[1])
	case "video":
		prompt, err = runtime.projection.BuildVideoPrompt(ctx, runtime.runner, pictures, input.Text[0], input.Text[1], input.FPS, false)
	case "mixed":
		prompt, err = runtime.projection.BuildMediaHistoryPrompt(ctx, runtime.runner, ordered, input.Text)
	}
	if err != nil {
		return nil, "", err
	}
	ids, projected, err := inference.CompileProjectedInputs(prompt, runtime.runner.Spec().EmbeddingLength)
	if err != nil {
		return nil, "", err
	}
	options.PromptTokenIDs, options.ProjectedInputs = ids, &projected
	return runtime.runner.Generate(ctx, "", options)
}

func verifyProjection(repository, path, projectorPath, input string,
	override modelintake.SessionOverride, residency recipe.ResidencyPolicy,
) error {
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	var suite evaluation.ExactSuite
	if err := json.Unmarshal([]byte(input), &suite); err != nil {
		return err
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return err
	}
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	_, definition, err := prepareCapability(ctx, store, path, recipe.TaskProjection, mediacapability.Projection(ctx, store, projectorPath))
	if err != nil {
		return err
	}
	if _, published, err := modelrecipe.Status(ctx, store, definition.ID); err != nil {
		return err
	} else if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "recipe/candidate/"+definition.ID.String(), definition); err != nil {
			return err
		}
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		return err
	}
	if _, err := modelrecipe.CompileCandidateExecution(ctx, store, program); err != nil {
		return err
	}
	candidate, err := prepareExactCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	_, _, declaredProcessor, err := projector.InspectProjection(ctx, projectorPath)
	if err != nil {
		return err
	}
	processor, err := projector.ResolvePreprocessProfile(ctx, store, definition, declaredProcessor)
	if err != nil {
		return err
	}
	projection, err := projector.OpenSession(ctx, projectorPath, projector.OpenOptions{CUDA: true, MediaPreprocess: processor})
	if err != nil {
		return err
	}
	if err := requireProjectionCoverage(suite, projection.Capabilities()); err != nil {
		return errors.Join(err, projection.Close())
	}
	runtime := &deferredExactRuntime{release: projection.Close, open: func(ctx context.Context) (exactRuntime, error) {
		runner, err := openExactCandidate(ctx, path, candidate)
		if err != nil {
			return nil, err
		}
		return &projectedExactRuntime{runner: runner, projection: projection}, nil
	}}
	return verifyExactRuntime(ctx, store, definition, revision, suite, exactPlan, candidate.Resolved.Document.ID, runtime)
}

type exactRuntime interface {
	evaluation.Generator
	Close() error
}

// The exact ledger calls Generate only for missing cases. Prepared resources
// transfer to the acquired runtime; otherwise Close releases them directly.
type deferredExactRuntime struct {
	open    func(context.Context) (exactRuntime, error)
	release func() error
	runtime exactRuntime
	closed  bool
}

// Generate acquires the language runtime once for missing case execution.
func (runtime *deferredExactRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if err := runtime.acquire(ctx); err != nil {
		return nil, "", err
	}
	return runtime.runtime.Generate(ctx, prompt, options)
}

func (runtime *deferredExactRuntime) acquire(ctx context.Context) error {
	if runtime.closed {
		return errors.New("recipe: exact runtime is closed")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if runtime.runtime == nil {
		opened, err := runtime.open(ctx)
		if err != nil {
			return err
		}
		if opened == nil {
			return errors.New("recipe: exact runtime acquisition returned nil")
		}
		runtime.runtime = opened
	}
	return nil
}

// Close releases acquired or prepared resources once.
func (runtime *deferredExactRuntime) Close() error {
	if runtime.closed {
		return nil
	}
	runtime.closed = true
	if runtime.runtime != nil {
		return runtime.runtime.Close()
	}
	if runtime.release != nil {
		return runtime.release()
	}
	return nil
}

// verifyExactRuntime shares real execution, failure publication and release for
// language-only and projected generation. Compilation alone cannot publish success.
func verifyExactRuntime(ctx context.Context, store *overgodb.Store, definition recipe.Definition, revision string,
	suite evaluation.ExactSuite, exactPlan evaluation.ExactPlan, modelDefinition artifact.ID, runtime *deferredExactRuntime,
) error {
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return errors.Join(err, runtime.Close())
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: modelDefinition,
		RuntimeRecipe:   definition.ID,
		CodeCommit:      revision,
		Environment:     environment.ID,
		Execution:       evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return errors.Join(fmt.Errorf("recipe: bind exact evaluation: %w", err), runtime.Close())
	}
	_, found, err := artifact.ReadContent(ctx, store, environment.ID)
	if err == nil && !found {
		var batch artifact.Batch
		batch, err = environment.Batch("recipe/exact-environment/" + environment.ID.String())
		if err == nil {
			_, err = artifact.CommitBatch(ctx, store, batch)
		}
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return errors.Join(err, runtime.Close())
	}
	started := time.Now()
	retained, retainedReport, err := retainedExactVerification(ctx, store, definition.ID, revision, environment.ID, evaluationPlan.Identity())
	if err != nil {
		return errors.Join(err, runtime.Close())
	}
	if retained.Gate.ID != (artifact.ID{}) {
		// A terminal record proves the previous lifecycle, including cleanup.
		// Missing ledger entries contradict it; never silently acquire again.
		runtime.open = func(context.Context) (exactRuntime, error) {
			return nil, errors.New("recipe: terminal verification has incomplete case evidence")
		}
	} else if err := runtime.acquire(ctx); err != nil {
		return errors.Join(err, runtime.Close())
	}
	reportID, evaluateErr := evaluation.EvaluateExactSharded(ctx, store, runtime, exactPlan, evaluationPlan, nil)
	closeErr := runtime.Close()
	if retained.Gate.ID != (artifact.ID{}) {
		if closeErr != nil || reportID != retainedReport {
			return errors.Join(evaluateErr, closeErr, errors.New("recipe: retained verification could not be replayed exactly"))
		}
		if retained.Gate.Outcome == runrecord.OutcomeFailed {
			evaluateErr = errors.Join(evaluateErr, fmt.Errorf("recipe: retained verification failed: %s", retained.Gate.Steps[0].Evidence))
		}
		return errors.Join(evaluateErr, json.NewEncoder(os.Stdout).Encode(map[string]any{
			"gate_id": retained.Gate.ID.String(), "recipe_id": definition.ID.String(),
			"run_id": retained.Run.ID.String(), "report_id": reportID.String(),
			"outcome": retained.Gate.Outcome, "reused": true,
		}))
	}
	if evaluateErr != nil || closeErr != nil {
		failure := errors.Join(evaluateErr, closeErr)
		evidence := fmt.Sprintf("plan=%s;report=%s; %s", evaluationPlan.Identity(), reportID, strings.ReplaceAll(failure.Error(), "\n", " "))
		if len(evidence) > 1900 {
			evidence = evidence[:1900]
		}
		verification, publishErr := modelintake.PublishFailure(
			ctx, store, definition, revision, time.Since(started),
			"cuda:0", "cuda", "contract=exact; "+evidence, "exact-mismatch",
		)
		if publishErr != nil {
			return errors.Join(failure, publishErr)
		}
		if encodeErr := json.NewEncoder(os.Stdout).Encode(map[string]any{
			"error": failure.Error(), "gate_id": verification.Gate.String(),
			"outcome": "failed", "recipe_id": definition.ID.String(),
			"run_id":    verification.Run.String(),
			"report_id": reportID.String(),
		}); encodeErr != nil {
			return errors.Join(failure, encodeErr)
		}
		return fmt.Errorf("recipe: exact generation evaluation: %w", failure)
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, revision, time.Since(started), "cuda:0", "cuda", evidence,
	)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"gate_id": verification.Gate.String(), "recipe_id": definition.ID.String(),
		"run_id": verification.Run.String(), "report_id": reportID.String(),
	})
}
