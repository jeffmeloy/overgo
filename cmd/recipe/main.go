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
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/seq2seq"
	"overgo/internal/seriesforecast"
	"overgo/internal/speechsynth"
	"overgo/internal/tabularicl"
)

type capabilityCommand struct {
	inventory func(string) (modelartifact.Inventory, error)
	execute   capabilityruntime.Executor
}

var capabilityCommands = map[recipe.Task]capabilityCommand{
	recipe.TaskForecast: {
		inventory: hfInventory,
		execute: capabilityruntime.JSONScalar[[]float32, *seriesforecast.Model, []float32](
			"forecast", seriesforecast.ValidateRequest,
			capabilityruntime.IgnoreInput[[]float32](seriesforecast.Load), seriesforecast.RegisterRuntime),
	},
	recipe.TaskTabular: {
		inventory: tabularInventory,
		execute: capabilityruntime.JSONScalar[tabularicl.Request, *tabularicl.Model, tabularicl.Prediction](
			"tabular", tabularicl.ValidateRequest,
			func(path string, request tabularicl.Request) (*tabularicl.Model, error) {
				return tabularicl.LoadTask(path, request.Task)
			}, tabularicl.RegisterRuntime),
	},
	recipe.TaskSeq2Seq: {
		inventory: hfInventory,
		execute: capabilityruntime.JSONScalar[seq2seq.GenerateRequest, *seq2seq.Model, []int](
			"seq2seq", seq2seq.ValidateGenerateRequest,
			capabilityruntime.IgnoreInput[seq2seq.GenerateRequest](seq2seq.Load), seq2seq.RegisterRuntime),
	},
	recipe.TaskSpeech: {
		inventory: speechInventory,
		execute: capabilityruntime.JSONScalar[speechsynth.SynthesisRequest, *speechsynth.Synthesizer, speechsynth.Audio](
			"speech", speechsynth.ValidateSynthesisRequest,
			capabilityruntime.IgnoreInput[speechsynth.SynthesisRequest](speechsynth.LoadSynthesizer), speechsynth.RegisterRuntime),
	},
	recipe.TaskImageGen: {
		inventory: imageGenInventory,
		execute: capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, oscillatorimage.Image](
			"image-gen", oscillatorimage.ValidateRequest,
			capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime),
	},
	recipe.TaskVideoGen: {
		inventory: videoGenInventory,
	},
	recipe.TaskVQA: {
		inventory: hfInventory,
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
	roots, err := dataroot.ResolveCurrent()
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

// videoGenInventory: denoiser, VAE, and text-encoder facts.
func videoGenInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("video-gen", path, "config.json",
		modelartifact.FileSpec{Path: "Wan2.1_VAE.pth", Name: "vae/weights", Role: artifact.ComponentWeights},
		modelartifact.FileSpec{Path: "models_t5_umt5-xxl-enc-bf16.pth", Name: "textenc/weights", Role: artifact.ComponentWeights})
}

// singleSafetensors: unique top-level weights file.
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

func safetensorsInventory(
	context, path, config string,
	companions ...modelartifact.FileSpec,
) (modelartifact.Inventory, error) {
	weights, err := singleSafetensors(context, path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	specs := []modelartifact.FileSpec{
		{Path: filepath.Join(path, config), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, weights), Name: "weights", Role: artifact.ComponentWeights},
	}
	for _, companion := range companions {
		companion.Path = filepath.Join(path, companion.Path)
		specs = append(specs, companion)
	}
	return modelartifact.FromFiles(path, specs)
}

// speechInventory: model, codec, and tokenizer facts.
func speechInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("speech", path, "pockettts_config.json",
		modelartifact.FileSpec{Path: "tokenizer.model", Name: "tokenizer", Role: artifact.ComponentTokenizer})
}

// imageGenInventory: config and weights facts.
func imageGenInventory(path string) (modelartifact.Inventory, error) {
	return safetensorsInventory("image-gen", path, "config.json")
}

// tabularInventory: classification and regression head facts.
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
	definition, err := modelrecipe.CapabilityDefinition(task, modelID)
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
