package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/cuda/driver"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/server"
	"overgo/internal/strictjson"
	"overgo/internal/tokenizer"
)

// verifyInference verifies the inference recipe the code derives for the
// model by the exact suite the input holds, or with protocol by the cases it lists.
func verifyInference(
	repository, path, input string,
	override modelintake.SessionOverride,
	residency recipe.ResidencyPolicy,
	protocol bool,
) error {
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	var suite evaluation.ExactSuite
	var exactPlan evaluation.ExactPlan
	if !protocol {
		if err := json.Unmarshal([]byte(input), &suite); err != nil {
			return fmt.Errorf("recipe: decode inference suite: %w", err)
		}
		if exactPlan, err = evaluation.CompileExact(suite); err != nil {
			return fmt.Errorf("recipe: compile inference suite: %w", err)
		}
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
	open := func(ctx context.Context) (exactRuntime, error) {
		return openExactCandidate(ctx, path, candidate)
	}
	if protocol {
		_, err = acceptProtocol(ctx, store, candidate.Definition, candidate.Resolved.Document.ID, revision, input, open)
	} else {
		_, err = verifyExactRuntime(ctx, store, candidate.Definition, revision, suite, exactPlan, candidate.Resolved.Document.ID, &deferredExactRuntime{open: open})
	}
	return err
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
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved, candidate.Inventory.Claim)
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

// projectionReverification is the decision a re-verification of an active
// projection recipe records.
const projectionReverification = "an exact suite ran again under the active projection recipe and matched every case"

// verifyProjection runs an exact suite through the projection recipe the code
// derives for the model and projector, and with reverify re-verifies the
// model's active recipe on the verification it records, as a training run
// re-verifies its recipe: the activation's accepted proof is then bound to the
// commit that ran it. A derived recipe that is not the active one is refused,
// since re-verification adopts no recipe and activating another is a reviewed
// decision.
func verifyProjection(repository, path, projectorPath, input string,
	override modelintake.SessionOverride, residency recipe.ResidencyPolicy, reverify bool,
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
	if reverify {
		active, found, err := modelrecipe.ActiveRecord(ctx, store, definition.Model, recipe.TaskProjection)
		if err != nil {
			return err
		}
		if !found || active.Definition.ID != definition.ID {
			return fmt.Errorf("recipe: the code derives %s, which is not the model's active projection recipe: re-verification adopts no recipe", definition.ID)
		}
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
	verification, err := verifyExactRuntime(ctx, store, definition, revision, suite, exactPlan, candidate.Resolved.Document.ID, runtime)
	if err != nil || !reverify {
		return err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, projectionReverification); err != nil {
		return err
	}
	fmt.Printf("projection recipe re-verified and active: %s (run %s at %.12s)\n", definition.ID, verification.Run, revision)
	return nil
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

// bindExactPlan binds an exact plan to the candidate recipe, its model's definition, the code and this machine's
// environment, which it publishes where the store lacks it.
func bindExactPlan(ctx context.Context, store *overgodb.Store, definition recipe.Definition, revision string,
	exactPlan evaluation.ExactPlan, modelDefinition artifact.ID,
) (evaluation.Plan, runrecord.Environment, error) {
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return evaluation.Plan{}, environment, err
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: modelDefinition,
		RuntimeRecipe:   definition.ID,
		CodeCommit:      revision,
		Environment:     environment.ID,
		Execution:       evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return evaluation.Plan{}, environment, fmt.Errorf("recipe: bind exact evaluation: %w", err)
	}
	_, found, err := artifact.ReadContent(ctx, store, environment.ID)
	if err == nil && !found {
		var batch artifact.Batch
		batch, err = environment.Batch("recipe/exact-environment/" + environment.ID.String())
		if err == nil {
			_, err = artifact.Publish(ctx, store, batch)
		}
	}
	return evaluationPlan, environment, err
}

// verifyExactRuntime shares real execution, failure publication and release for
// language-only and projected generation. Compilation alone cannot publish
// success. It returns the verification it published or replayed, which
// succeeded when it returns no error.
func verifyExactRuntime(ctx context.Context, store *overgodb.Store, definition recipe.Definition, revision string,
	suite evaluation.ExactSuite, exactPlan evaluation.ExactPlan, modelDefinition artifact.ID, runtime *deferredExactRuntime,
) (modelrecipe.Verification, error) {
	evaluationPlan, environment, err := bindExactPlan(ctx, store, definition, revision, exactPlan, modelDefinition)
	if err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	started := time.Now()
	retained, retainedReport, err := retainedExactVerification(ctx, store, definition.ID, revision, environment.ID, evaluationPlan.Identity())
	if err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	if retained.Gate.ID != (artifact.ID{}) {
		// A terminal record proves the previous lifecycle, including cleanup.
		// Missing ledger entries contradict it; never silently acquire again.
		runtime.open = func(context.Context) (exactRuntime, error) {
			return nil, errors.New("recipe: terminal verification has incomplete case evidence")
		}
	} else if err := runtime.acquire(ctx); err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	reportID, evaluateErr := evaluation.EvaluateExactSharded(ctx, store, runtime, exactPlan, evaluationPlan, nil)
	closeErr := runtime.Close()
	if retained.Gate.ID != (artifact.ID{}) {
		if closeErr != nil || reportID != retainedReport {
			return modelrecipe.Verification{}, errors.Join(evaluateErr, closeErr, errors.New("recipe: retained verification could not be replayed exactly"))
		}
		if retained.Gate.Outcome == runrecord.OutcomeFailed {
			evaluateErr = errors.Join(evaluateErr, fmt.Errorf("recipe: retained verification failed: %s", retained.Gate.Steps[0].Evidence))
		}
		return modelrecipe.Verification{Gate: retained.Gate.ID, Run: retained.Run.ID}, errors.Join(evaluateErr, json.NewEncoder(os.Stdout).Encode(map[string]any{
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
			return modelrecipe.Verification{}, errors.Join(failure, publishErr)
		}
		if encodeErr := json.NewEncoder(os.Stdout).Encode(map[string]any{
			"error": failure.Error(), "gate_id": verification.Gate.String(),
			"outcome": "failed", "recipe_id": definition.ID.String(),
			"run_id":    verification.Run.String(),
			"report_id": reportID.String(),
		}); encodeErr != nil {
			return modelrecipe.Verification{}, errors.Join(failure, encodeErr)
		}
		return modelrecipe.Verification{}, fmt.Errorf("recipe: exact generation evaluation: %w", failure)
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, revision, time.Since(started), "cuda:0", "cuda", evidence,
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	return verification, json.NewEncoder(os.Stdout).Encode(map[string]any{
		"gate_id": verification.Gate.String(), "recipe_id": definition.ID.String(),
		"run_id": verification.Run.String(), "report_id": reportID.String(),
	})
}

// protocolObservation is the schema of what a protocol acceptance observed:
// each case it sent, with what the served model gave for it.
const protocolObservation = "overgo/protocol-acceptance/v1"

// protocolCase is one request a protocol acceptance sends through the serving
// handlers, with the answer an independent reader expects of it, and what
// came back. A case that cancels hangs up once streamed content reaches it.
type protocolCase struct {
	Name     string          `json:"name"`
	Route    string          `json:"route"`
	Body     json.RawMessage `json:"body"`
	Expected string          `json:"expected,omitzero"`
	Cancel   bool            `json:"cancel,omitzero"`
	Status   int             `json:"status,omitzero"`
	Answer   string          `json:"answer,omitzero"`
	Finish   string          `json:"finish,omitzero"`
	Accepted bool            `json:"accepted,omitzero"`
	Response string          `json:"response,omitzero"`
}

// acceptProtocol sends the listed cases through the serving handlers of the
// runtime open loads. Only when every case is accepted, the runtime closes and
// the process then holds no device memory does it publish the verification,
// whose run binds the observation as its input; otherwise it publishes nothing.
// It also records the first chat case as an exact plan on the same runtime and
// names the plan in the gate's evidence, so recipe measure can replay it.
func acceptProtocol(ctx context.Context, store *overgodb.Store, definition recipe.Definition, modelDefinition artifact.ID,
	revision, input string, open func(context.Context) (exactRuntime, error),
) (modelrecipe.Verification, error) {
	var cases []protocolCase
	if err := strictjson.DecodeBytes([]byte(input), &cases); err != nil || len(cases) == 0 {
		return modelrecipe.Verification{}, errors.Join(errors.New("recipe: the protocol input lists no cases"), err)
	}
	started := time.Now()
	runtime, err := open(ctx)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	handler, err := server.New(server.Config{MaxConcurrent: 1}, runtime)
	if err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	var failure error
	for index := range cases {
		if test := &cases[index]; !serveProtocolCase(ctx, handler, test) {
			failure = errors.Join(failure, fmt.Errorf("recipe: protocol case %s answered %q, finishing %q", test.Name, test.Answer, test.Finish))
		}
	}
	var planID, reportID artifact.ID
	if failure == nil {
		planID, reportID, failure = bindProtocolExact(ctx, store, definition, modelDefinition, revision, runtime, cases)
	}
	failure = errors.Join(failure, handler.Close(), runtime.Close())
	if held := driver.ProcessCurrentBytes(); held != 0 {
		failure = errors.Join(failure, fmt.Errorf("recipe: the process holds %d device bytes after close", held))
	}
	observation, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, protocolObservation), cases)
	failure = errors.Join(failure, err)
	result := map[string]any{"recipe_id": definition.ID.String(), "observation_id": observation.Descriptor.ID.String(), "cases": cases}
	var verification modelrecipe.Verification
	if failure == nil {
		verification, failure = modelintake.PublishMeasuredVerification(ctx, store, definition, revision, time.Since(started), "cuda:0", "cuda",
			fmt.Sprintf("contract=protocol;observation=%s;cases=%d;plan=%s;report=%s; every case answered as expected through the serving handlers, the runtime closed and the process held no device bytes after", observation.Descriptor.ID, len(cases), planID, reportID),
			capabilityruntime.Measured{Input: observation})
		result["gate_id"], result["run_id"] = verification.Gate.String(), verification.Run.String()
	}
	return verification, errors.Join(failure, json.NewEncoder(os.Stdout).Encode(result))
}

// protocolChatRoute is the route whose cases bind the exact plan a protocol acceptance names.
const protocolChatRoute = "/v1/chat/completions"

// bindProtocolExact records a protocol acceptance's first chat case on the runtime as an exact case, its
// conversation formatted as the served model formats a chat under that case's token budget, and evaluates the
// suite it makes on the same runtime: the plan and report recipe measure replays.
func bindProtocolExact(ctx context.Context, store *overgodb.Store, definition recipe.Definition, modelDefinition artifact.ID,
	revision string, runtime exactRuntime, cases []protocolCase,
) (plan, report artifact.ID, err error) {
	formatter, ok := runtime.(interface {
		FormatChatWithOptions([]inference.ChatMessage, inference.ChatFormatOptions) (string, error)
	})
	if !ok {
		return plan, report, errors.New("recipe: the served model formats no chat to bind as an exact plan")
	}
	for _, test := range cases {
		var body struct {
			Messages  []inference.ChatMessage `json:"messages"`
			MaxTokens int                     `json:"max_tokens"`
		}
		if test.Route != protocolChatRoute || test.Cancel || json.Unmarshal(test.Body, &body) != nil ||
			len(body.Messages) == 0 || body.MaxTokens <= 0 {
			continue
		}
		prompt, err := formatter.FormatChatWithOptions(body.Messages, inference.ChatFormatOptions{AddGenerationPrompt: true})
		if err != nil {
			return plan, report, err
		}
		recorded, err := evaluation.Record(ctx, runtime, test.Name, prompt, body.MaxTokens)
		if err != nil {
			return plan, report, err
		}
		suite := evaluation.ExactSuite{
			Schema: "overgo/protocol-exact/v1", Source: "the protocol acceptance's first chat case, recorded on its candidate",
			Cases: []evaluation.ExactCase{{
				Name: test.Name, Prompt: prompt, MaxTokens: body.MaxTokens, Text: recorded.Text,
				PromptTokens: recorded.PromptTokens, GeneratedTokens: recorded.GeneratedTokens,
			}},
		}
		exactPlan, err := evaluation.CompileExact(suite)
		if err != nil {
			return plan, report, err
		}
		evaluationPlan, _, err := bindExactPlan(ctx, store, definition, revision, exactPlan, modelDefinition)
		if err != nil {
			return plan, report, err
		}
		report, err = evaluation.EvaluateExactSharded(ctx, store, runtime, exactPlan, evaluationPlan, nil)
		return evaluationPlan.Identity(), report, err
	}
	return plan, report, errors.New("recipe: the protocol input lists no chat case to bind as an exact plan")
}

// protocolEvidencePlan reads the exact plan a protocol verification's evidence names, zero where it names none.
func protocolEvidencePlan(evidence string) artifact.ID {
	fields := strings.Split(evidence, ";")
	if strings.TrimSpace(fields[0]) != "contract=protocol" {
		return artifact.ID{}
	}
	for _, field := range fields[1:] {
		if text, declared := strings.CutPrefix(strings.TrimSpace(field), "plan="); declared {
			plan, _ := artifact.ParseID(text)
			return plan
		}
	}
	return artifact.ID{}
}

// serveProtocolCase sends one case through the handler and records what came
// back. The answer is the text of whichever protocol's fields hold it, whole
// or streamed, and a case is accepted when it is the expected answer of a
// finished turn; a cancelled case is accepted when its turn did not finish.
func serveProtocolCase(ctx context.Context, handler http.Handler, test *protocolCase) bool {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	request := httptest.NewRequestWithContext(ctx, http.MethodPost, test.Route, bytes.NewReader(test.Body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	client := http.ResponseWriter(recorder)
	if test.Cancel {
		client = hangUp{recorder, cancel}
	}
	handler.ServeHTTP(client, request)
	test.Status, test.Response, test.Answer, test.Finish = recorder.Code, recorder.Body.String(), "", ""
	payloads := []string{test.Response}
	stream := recorder.Header().Get("Content-Type") == "text/event-stream"
	if stream {
		payloads = nil
		for line := range strings.Lines(test.Response) {
			if payload, found := strings.CutPrefix(strings.TrimSpace(line), "data: "); found && payload != "[DONE]" {
				payloads = append(payloads, payload)
			}
		}
	}
	parsed := !stream || strings.Contains(test.Response, "data: [DONE]")
	for _, payload := range payloads {
		var turn protocolTurn
		parsed = json.Unmarshal([]byte(payload), &turn) == nil && parsed
		answer, finish := turn.answer()
		test.Answer, test.Finish = test.Answer+answer, cmp.Or(finish, test.Finish)
	}
	test.Accepted = parsed && test.Status == http.StatusOK && test.Finish == "stop" && strings.TrimSpace(test.Answer) == test.Expected
	if test.Cancel {
		test.Accepted = ctx.Err() != nil && test.Finish == ""
	}
	return test.Accepted
}

// hangUp is a client that hangs up once streamed content reaches it.
type hangUp struct {
	*httptest.ResponseRecorder
	cancel context.CancelFunc
}

// Flush delivers what was written, and hangs up once that holds content.
func (client hangUp) Flush() {
	client.ResponseRecorder.Flush()
	if strings.Contains(client.Body.String(), `"content":`) {
		client.cancel()
	}
}

// protocolTurn holds the fields a turn's answer and finish come in under the
// chat, responses and messages protocols.
type protocolTurn struct {
	Choices []struct {
		Message struct {
			Content   string
			ToolCalls []inference.ChatToolCall `json:"tool_calls"`
		}
		Delta        struct{ Content string }
		FinishReason string `json:"finish_reason"`
	}
	Output     []struct{ Content []struct{ Text string } }
	Content    []struct{ Text string }
	Status     string
	StopReason string `json:"stop_reason"`
}

// answer is the turn's text and how it finished, stop for a finished turn
// under every protocol. A turn that ends in one call of a tool with one
// argument answers the tool's name and the argument's value.
func (turn protocolTurn) answer() (string, string) {
	text, finish := "", ""
	for _, choice := range turn.Choices {
		text, finish = text+choice.Message.Content+choice.Delta.Content, cmp.Or(choice.FinishReason, finish)
		var arguments map[string]string
		if calls := choice.Message.ToolCalls; len(calls) == 1 && finish == "tool_calls" &&
			json.Unmarshal([]byte(calls[0].Function.Arguments), &arguments) == nil && len(arguments) == 1 {
			for _, value := range arguments {
				return calls[0].Function.Name + ":" + value, "stop"
			}
		}
	}
	for _, item := range turn.Output {
		for _, part := range item.Content {
			text += part.Text
		}
	}
	for _, part := range turn.Content {
		text += part.Text
	}
	if turn.Status == "completed" || turn.StopReason == "end_turn" {
		finish = "stop"
	}
	return text, finish
}
