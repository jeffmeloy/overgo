// recipe: drives the model-recipe lifecycle for real artifacts — the command
// the sealed authority was waiting for (the lifecycle API was complete but
// uninvoked; the wave-1 probe recorded the refusal that proved it).
//
//	recipe activate -reason "..." <model>   publish facts + definition,
//	                                        candidate -> validated -> active
//	recipe status <model>                   show the active recipe and tier
//
// Model references resolve through the data-root contract; the store is the
// data-root store unless -repo overrides. Activation records its reason and
// decider commit in the decision event, per the store's decision discipline.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/strictjson"
	"overgo/internal/tabularicl"
	"overgo/internal/workflowruntime"
)

var forecastInputContract = artifact.JSONContract(artifact.KindFile, "overgo.forecast-input.v1")

var tabularInputContract = artifact.JSONContract(artifact.KindFile, "overgo.tabular-input.v1")

var seq2seqInputContract = artifact.JSONContract(artifact.KindFile, "overgo.seq2seq-input.v1")

var speechInputContract = artifact.JSONContract(artifact.KindFile, "overgo.speech-input.v1")

type capabilityExecutor func(
	context.Context,
	artifact.Repository,
	string,
	artifact.ID,
	recipe.Program,
	string,
) (any, error)

type capabilityCommand struct {
	inventory  func(string) (modelartifact.Inventory, error)
	definition func(artifact.ID) (recipe.Definition, error)
	execute    capabilityExecutor
}

var capabilityCommands = map[recipe.Task]capabilityCommand{
	recipe.TaskForecast: {
		inventory: hfInventory, definition: modelrecipe.ForecastDefinition, execute: executeForecast,
	},
	recipe.TaskTabular: {
		inventory: tabularInventory, definition: modelrecipe.TabularDefinition, execute: executeTabular,
	},
	recipe.TaskSeq2Seq: {
		inventory: hfInventory, definition: modelrecipe.Seq2SeqDefinition, execute: executeSeq2Seq,
	},
	recipe.TaskSpeech: {
		inventory: speechInventory, definition: modelrecipe.SpeechDefinition, execute: executeSpeech,
	},
	recipe.TaskImageGen: {
		inventory: imageGenInventory, definition: modelrecipe.ImageGenDefinition,
	},
	recipe.TaskVideoGen: {
		inventory: videoGenInventory, definition: modelrecipe.VideoGenDefinition,
	},
	recipe.TaskVQA: {
		inventory: hfInventory, definition: modelrecipe.VQADefinition,
	},
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "recipe: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: recipe <activate|run|status> [options] <model>")
	}
	verb := os.Args[1]
	flags := flag.NewFlagSet("recipe "+verb, flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "RepoDB store; empty resolves via the data-root contract")
	reason := flags.String("reason", "", "activation reason recorded in the decision event (activate)")
	task := flags.String("task", string(recipe.TaskInference), "recipe task (inference|forecast|tabular|seq2seq|speech|image-gen|video-gen|vqa)")
	sessionFlag := flags.String("session", "auto", "decode session: auto (derive from plan) | request | capacity")
	input := flags.String("input", "", "task input as JSON (run)")
	if err := flags.Parse(os.Args[2:]); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("exactly one model reference is required")
	}
	working, err := os.Getwd()
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(working)
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		repository = roots.Store
	}
	path := roots.ResolveModelPath(flags.Arg(0))
	selectedTask := recipe.Task(*task)
	capability, capabilityKnown := capabilityCommands[selectedTask]
	switch verb {
	case "activate":
		if strings.TrimSpace(*reason) == "" {
			return errors.New("activate requires -reason: the decision event records why")
		}
		if capabilityKnown {
			return activateCapability(repository, path, *reason, selectedTask, capability)
		}
		if selectedTask != recipe.TaskInference {
			return fmt.Errorf("unsupported model recipe task %q", selectedTask)
		}
		sessionOverride, sessionErr := parseSessionOverride(*sessionFlag)
		if sessionErr != nil {
			return sessionErr
		}
		return activate(repository, path, *reason, sessionOverride)
	case "run":
		if !capabilityKnown || capability.execute == nil {
			return fmt.Errorf("task %q has no registered runtime", selectedTask)
		}
		if strings.TrimSpace(*input) == "" {
			return errors.New("run requires -input JSON")
		}
		return executeCapability(repository, path, selectedTask, capability, *input)
	case "status":
		return status(repository, path, selectedTask)
	default:
		return fmt.Errorf("unknown verb %q", verb)
	}
}

func hfInventory(path string) (modelartifact.Inventory, error) {
	repository, err := hfrepo.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	defer repository.Close()
	return modelartifact.FromHFRepository(repository)
}

// videoGenInventory: the text-to-video artifact is an explicit four-part
// list -- diffusion config + denoiser weights (the one top-level
// safetensors), the VAE checkpoint, and the text-encoder checkpoint (both
// pytorch-zip; read by internal/pytorchzip). The tokenizer directory and
// vendor repo are companions, not components.
func videoGenInventory(path string) (modelartifact.Inventory, error) {
	weights, err := singleSafetensors("video-gen", path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "Wan2.1_VAE.pth"), Name: "vae/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "models_t5_umt5-xxl-enc-bf16.pth"), Name: "textenc/weights", Role: artifact.ComponentWeights},
	})
}

// singleSafetensors: the ONE top-level .safetensors weights file of a
// capability artifact. Vendor-chosen names (pocket-tts content hash, un0
// model.safetensors, u-vit step_NNN.safetensors) are discovered, never
// assumed; more than one is ambiguous and refused.
func singleSafetensors(context, path string) (string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", err
	}
	weights := ""
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
			continue
		}
		if weights != "" {
			return "", fmt.Errorf("%s inventory: multiple safetensors files in %s", context, path)
		}
		weights = entry.Name()
	}
	if weights == "" {
		return "", fmt.Errorf("%s inventory: no safetensors weights in %s", context, path)
	}
	return weights, nil
}

// speechInventory: the pocket-tts layout is an explicit file list — the
// vendor-resolved pockettts_config.json, the discovered weights file, and
// the tokenizer.model the text conditioner reads. The embeddings/clips
// subdirectories are voice data, not model components.
func speechInventory(path string) (modelartifact.Inventory, error) {
	weights, err := singleSafetensors("speech", path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "pockettts_config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "tokenizer.model"), Name: "tokenizer", Role: artifact.ComponentTokenizer},
	})
}

// imageGenInventory: image-gen artifacts are an explicit two-component list
// — config.json (family tag / execution facts) plus the discovered weights
// file; every dimension derives from tensor lengths. Vendor scripts, sample
// renders, and provenance sidecars are not model components.
func imageGenInventory(path string) (modelartifact.Inventory, error) {
	weights, err := singleSafetensors("image-gen", path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
	})
}

// tabularInventory: dual-head artifact (classification/ + regression/, each
// config.json + model.safetensors) — an explicit file list; no repository
// walker owns this layout.
func tabularInventory(path string) (modelartifact.Inventory, error) {
	var specs []modelartifact.FileSpec
	for _, head := range tabularicl.Tasks() {
		specs = append(specs,
			modelartifact.FileSpec{
				Path: filepath.Join(path, head, "config.json"),
				Name: head + "/config", Role: artifact.ComponentConfig,
			},
			modelartifact.FileSpec{
				Path: filepath.Join(path, head, "model.safetensors"),
				Name: head + "/weights", Role: artifact.ComponentWeights,
			},
		)
	}
	return modelartifact.FromFiles(path, specs)
}

// activateCapability drives the shared sealed lifecycle for capability-package
// models: inventory facts, then candidate -> validated -> active with the
// decision evidence. Capability packages derive dimensions from the artifact,
// so no profile document is published.
func activateCapability(
	repository, path, reason string,
	task recipe.Task,
	capability capabilityCommand,
) error {
	ctx := context.Background()
	inventory, err := capability.inventory(path)
	if err != nil {
		return err
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID := inventory.Manifest.ID
	batch, err := inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	definition, err := capability.definition(modelID)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "recipe/candidate/"+definition.ID.String(), definition,
	); err != nil {
		return fmt.Errorf("publish candidate: %w", err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/validated/"+definition.ID.String(), definition,
		recipe.StatusValidated, nil, nil,
	); err != nil {
		return fmt.Errorf("transition validated: %w", err)
	}
	evidenceID, err := activationEvidence(ctx, store, definition, reason)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/active/"+definition.ID.String(), definition,
		recipe.StatusActive, []artifact.ID{evidenceID}, nil,
	); err != nil {
		return fmt.Errorf("transition active: %w", err)
	}
	fmt.Printf("activated %s\n  task       %s\n  model      %s\n  recipe     %s\n  reason     %s\n",
		path, task, modelID, definition.ID, reason)
	return nil
}

func executeCapability(
	repository, path string,
	task recipe.Task,
	capability capabilityCommand,
	input string,
) error {
	ctx := context.Background()
	inventory, err := capability.inventory(path)
	if err != nil {
		return err
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID := inventory.Manifest.ID
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, task)
	if err != nil {
		return err
	}
	if !active {
		return fmt.Errorf("model %s has no active %s recipe", modelID, task)
	}
	program, err := modelrecipe.CompileCapability(activation.Definition)
	if err != nil {
		return err
	}
	output, err := capability.execute(ctx, store, path, modelID, program, input)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(output)
}

func executeForecast(
	ctx context.Context,
	store artifact.Repository,
	path string,
	modelID artifact.ID,
	program recipe.Program,
	input string,
) (any, error) {
	series, content, err := forecastInput(input)
	if err != nil {
		return nil, err
	}
	model, err := seriesforecast.Load(path)
	if err != nil {
		return nil, err
	}
	return executeScalarProgram[[]float32](
		ctx, store, program, series, content,
		func(runtime *workflowruntime.Runtime) error {
			return seriesforecast.RegisterRuntime(runtime, modelID, model)
		},
	)
}

func executeTabular(
	ctx context.Context,
	store artifact.Repository,
	path string,
	modelID artifact.ID,
	program recipe.Program,
	input string,
) (any, error) {
	request, content, err := tabularInput(input)
	if err != nil {
		return nil, err
	}
	model, err := tabularicl.LoadTask(path, request.Task)
	if err != nil {
		return nil, err
	}
	return executeScalarProgram[tabularicl.Prediction](
		ctx, store, program, request, content,
		func(runtime *workflowruntime.Runtime) error {
			return tabularicl.RegisterRuntime(runtime, modelID, model)
		},
	)
}

func executeSeq2Seq(
	ctx context.Context,
	store artifact.Repository,
	path string,
	modelID artifact.ID,
	program recipe.Program,
	input string,
) (any, error) {
	request, content, err := seq2seqInput(input)
	if err != nil {
		return nil, err
	}
	model, err := seq2seq.Load(path)
	if err != nil {
		return nil, err
	}
	return executeScalarProgram[[]int](
		ctx, store, program, request, content,
		func(runtime *workflowruntime.Runtime) error {
			return seq2seq.RegisterRuntime(runtime, modelID, model)
		},
	)
}

func executeSpeech(
	ctx context.Context,
	store artifact.Repository,
	path string,
	modelID artifact.ID,
	program recipe.Program,
	input string,
) (any, error) {
	request, content, err := speechInput(input)
	if err != nil {
		return nil, err
	}
	synthesizer, err := speechsynth.LoadSynthesizer(path)
	if err != nil {
		return nil, err
	}
	return executeScalarProgram[speechsynth.Audio](
		ctx, store, program, request, content,
		func(runtime *workflowruntime.Runtime) error {
			return speechsynth.RegisterRuntime(runtime, modelID, synthesizer)
		},
	)
}

func executeScalarProgram[Output any](
	ctx context.Context,
	store artifact.Repository,
	program recipe.Program,
	value any,
	content artifact.Content,
	bind func(*workflowruntime.Runtime) error,
) (Output, error) {
	var zero Output
	definition := program.Definition()
	if len(definition.Inputs) != 1 || len(definition.Outputs) != 1 {
		return zero, errors.New("recipe command: scalar execution requires one input and output")
	}
	input, output := definition.Inputs[0], definition.Outputs[0]
	runtime, err := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
	if err != nil {
		return zero, err
	}
	if err := bind(runtime); err != nil {
		return zero, err
	}
	result, err := runtime.ExecuteProgram(
		ctx,
		"recipe/run/"+definition.ID.String()+"/"+content.Descriptor.ID.String(),
		program,
		map[recipe.PortName]workflowruntime.Value{
			input.Name: workflowruntime.ArtifactValue(input.Data, value, content),
		},
	)
	if err != nil {
		return zero, err
	}
	datum, ok := result.Outputs[output.Name].Single()
	if !ok {
		return zero, fmt.Errorf("runtime output %q has invalid cardinality", output.Name)
	}
	decoded, ok := datum.Value.(Output)
	if !ok {
		return zero, fmt.Errorf("runtime output %q has invalid value type", output.Name)
	}
	return decoded, nil
}

func forecastInput(input string) ([]float32, artifact.Content, error) {
	return decodeCapabilityInput(input, "forecast", forecastInputContract, validateForecastInput)
}

func validateForecastInput(series []float32) error {
	if len(series) == 0 {
		return errors.New("forecast input series is empty")
	}
	for _, value := range series {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("forecast input series contains a non-finite value")
		}
	}
	return nil
}

func tabularInput(input string) (tabularicl.Request, artifact.Content, error) {
	return decodeCapabilityInput(input, "tabular", tabularInputContract, tabularicl.ValidateRequest)
}

func seq2seqInput(input string) (seq2seq.GenerateRequest, artifact.Content, error) {
	return decodeCapabilityInput(input, "seq2seq", seq2seqInputContract, seq2seq.ValidateGenerateRequest)
}

func speechInput(input string) (speechsynth.SynthesisRequest, artifact.Content, error) {
	return decodeCapabilityInput(input, "speech", speechInputContract, speechsynth.ValidateSynthesisRequest)
}

func decodeCapabilityInput[Input any](
	raw, name string,
	contract artifact.DocumentContract,
	validate func(Input) error,
) (Input, artifact.Content, error) {
	var value Input
	if err := strictjson.DecodeBytes([]byte(raw), &value); err != nil {
		return value, artifact.Content{}, fmt.Errorf("decode %s input: %w", name, err)
	}
	if err := validate(value); err != nil {
		return value, artifact.Content{}, err
	}
	content, err := artifact.JSONContent(contract, value)
	return value, content, err
}

// sessionOverride: operator-pinned decode session; nil defers to the
// plan-derived choice.
type sessionOverride struct {
	set   bool
	value modelrecipe.DecodeSessionPolicy
}

func parseSessionOverride(text string) (sessionOverride, error) {
	switch strings.ToLower(strings.TrimSpace(text)) {
	case "", "auto":
		return sessionOverride{}, nil
	case "request":
		return sessionOverride{set: true, value: modelrecipe.DecodeSessionRequest}, nil
	case "capacity":
		return sessionOverride{set: true, value: modelrecipe.DecodeSessionCapacity}, nil
	default:
		return sessionOverride{}, fmt.Errorf("unknown -session %q (auto|request|capacity)", text)
	}
}

func activate(repository, path, reason string, override sessionOverride) error {
	ctx := context.Background()
	file, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		return err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return err
	}
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		return fmt.Errorf("architecture %q has no registered profile", spec.Architecture)
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		return err
	}
	document, err := modelrecipe.NewModelDefinitionFromGGUF(file, profileDocument, inventory.TensorInventory)
	if err != nil {
		return err
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return err
	}
	// session derived from the compiled plan: capacity requires every token
	// cache to admit bounded append; SharedKV models compile only as request.
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return err
	}
	modelPlan, err := model.CompileModelPlan(spec, weights)
	if err != nil {
		return err
	}
	session := modelrecipe.DecodeSessionCapacity
	if !modelPlan.SupportsCapacityCache() {
		session = modelrecipe.DecodeSessionRequest
	}
	if override.set {
		// operator pin: request is the general per-request path (valid for any
		// model); capacity requires SupportsCapacityCache.
		if override.value == modelrecipe.DecodeSessionCapacity && !modelPlan.SupportsCapacityCache() {
			return errors.New("recipe: -session capacity is unsupported by this model's compiled plan")
		}
		session = override.value
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID := inventory.Manifest.ID
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, "recipe/facts/"+modelID.String(), inventory, resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		session,
	)
	if err != nil {
		return err
	}
	// resumable lifecycle: pick up from wherever this definition already is
	state, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+definition.ID.String(), definition,
		); err != nil {
			return fmt.Errorf("publish candidate: %w", err)
		}
		state = recipe.StatusCandidate
	}
	if state == recipe.StatusCandidate {
		if _, _, err := modelrecipe.Transition(
			ctx, store, "recipe/validated/"+definition.ID.String(), definition,
			recipe.StatusValidated, nil, nil,
		); err != nil {
			return fmt.Errorf("transition validated: %w", err)
		}
		state = recipe.StatusValidated
	}
	switch state {
	case recipe.StatusValidated:
		evidenceID, err := activationEvidence(ctx, store, definition, reason)
		if err != nil {
			return err
		}
		// activation supersedes any current active recipe for this model+task
		var supersedes *artifact.ID
		if current, active, err := modelrecipe.ActiveRecord(
			ctx, store, modelID, recipe.TaskInference,
		); err == nil && active && current.Definition.ID != definition.ID {
			id := current.Definition.ID
			supersedes = &id
		}
		if _, _, err := modelrecipe.Transition(
			ctx, store, "recipe/active/"+definition.ID.String(), definition,
			recipe.StatusActive, []artifact.ID{evidenceID}, supersedes,
		); err != nil {
			return fmt.Errorf("transition active: %w", err)
		}
	case recipe.StatusActive:
	default:
		return fmt.Errorf("recipe %s is %q; activation resumes only from candidate or validated", definition.ID, state)
	}
	fmt.Printf("activated %s\n  model      %s\n  definition %s\n  recipe     %s\n  reason     %s\n",
		path, modelID, resolved.Document.ID, definition.ID, reason)
	return nil
}

// activationEvidence records the decision basis: reason, decider commit, and
// the honest statement that serving-parity evidence follows as run records.
func activationEvidence(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	reason string,
) (artifact.ID, error) {
	decider := "unknown"
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		decider = strings.TrimSpace(string(out))
	}
	payload := []byte("activation decision\nreason: " + reason + "\ndecider_commit: " + decider +
		"\nfollow_up: serving parity evidence lands as run records against this recipe\n")
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		return artifact.ID{}, err
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key: "recipe/activation-evidence/" + definition.ID.String(),
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload))},
			Data:       payload,
		}},
	})
	if err != nil {
		return artifact.ID{}, err
	}
	return id, nil
}

func status(repository, path string, task recipe.Task) error {
	ctx := context.Background()
	var inventory modelartifact.Inventory
	if capability, ok := capabilityCommands[task]; ok {
		var err error
		inventory, err = capability.inventory(path)
		if err != nil {
			return err
		}
	} else {
		if task != recipe.TaskInference {
			return fmt.Errorf("unsupported model recipe task %q", task)
		}
		file, err := gguf.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		inventory, err = modelartifact.FromGGUF(file)
		if err != nil {
			return err
		}
	}
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, inventory.Manifest.ID, task)
	if err != nil {
		return err
	}
	if !active {
		fmt.Printf("%s\n  model  %s\n  active %s recipe: ABSENT\n", path, inventory.Manifest.ID, task)
		return nil
	}
	fmt.Printf("%s\n  model  %s\n  recipe %s\n  tier   %s\n",
		path, inventory.Manifest.ID, activation.Definition.ID, activation.Tier)
	return nil
}
