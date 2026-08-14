// recipe: drives the model-recipe lifecycle for real artifacts — the command
// the sealed authority was waiting for (the lifecycle API was complete but
// uninvoked; the wave-1 probe recorded the refusal that proved it).
//
//	recipe activate -reason "..." -gate <id> -run-id <id> <model>
//	                                        verified promotion
//	recipe status <model>                   show the active recipe and tier
//
// Model references resolve through the data-root contract. Activation consumes
// a successful recipe-bound verifier gate/run pair.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

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
	gate := flags.String("gate", "", "successful verifier gate artifact ID (activate)")
	runID := flags.String("run-id", "", "bound verifier run artifact ID (activate)")
	task := flags.String("task", string(recipe.TaskInference), "recipe task (inference|forecast|tabular|seq2seq|speech|image-gen|vqa)")
	sessionFlag := flags.String("session", "auto", "decode session: auto (derive from plan) | request | capacity")
	residencyFlag := flags.String("residency", string(recipe.ResidencyHybridNative),
		"weight residency: stream | host-cache | device-f32 | device-native | device-native-bf16 | hybrid-native | host-reference")
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
	capability, capabilityKnown := capabilities[selectedTask]
	switch verb {
	case "activate":
		if strings.TrimSpace(*reason) == "" {
			return errors.New("activate requires -reason: the decision event records why")
		}
		verification, verificationErr := parseVerification(*gate, *runID)
		if verificationErr != nil {
			return verificationErr
		}
		if capabilityKnown {
			return activateCapability(repository, path, *reason, selectedTask, capability, verification)
		}
		if selectedTask != recipe.TaskInference {
			return fmt.Errorf("unsupported model recipe task %q", selectedTask)
		}
		sessionOverride, sessionErr := parseSessionOverride(*sessionFlag)
		if sessionErr != nil {
			return sessionErr
		}
		residency, residencyErr := parseResidency(*residencyFlag)
		if residencyErr != nil {
			return residencyErr
		}
		return activate(repository, path, *reason, sessionOverride, residency, verification)
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

func parseResidency(text string) (recipe.ResidencyPolicy, error) {
	policy := recipe.ResidencyPolicy(strings.ToLower(strings.TrimSpace(text)))
	if policy == "" || !policy.Valid() {
		return "", fmt.Errorf("unknown -residency %q", text)
	}
	return policy, nil
}

func parseVerification(gateText, runText string) (modelrecipe.Verification, error) {
	gate, err := artifact.ParseID(strings.TrimSpace(gateText))
	if err != nil || gate.Kind() != artifact.KindEvidence {
		return modelrecipe.Verification{}, errors.New("activate requires -gate with an evidence artifact ID")
	}
	run, err := artifact.ParseID(strings.TrimSpace(runText))
	if err != nil || run.Kind() != artifact.KindRun {
		return modelrecipe.Verification{}, errors.New("activate requires -run-id with a run artifact ID")
	}
	return modelrecipe.Verification{Gate: gate, Run: run}, nil
}

// activateCapability: capability facts, candidate, validation, verified promotion.
func activateCapability(
	repository, path, reason string,
	task recipe.Task,
	capability capability,
	verification modelrecipe.Verification,
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
	var definition recipe.Definition
	var facts []artifact.Content
	if capability.bind != nil {
		definition, facts, err = capability.bind(path, modelID)
	} else {
		definition, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	if err != nil {
		return err
	}
	batch, err := inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, facts...)
	if _, err := store.Commit(ctx, batch); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceExperimental, reason); err != nil {
		return err
	}
	fmt.Printf("activated %s\n  task       %s\n  model      %s\n  recipe     %s\n  reason     %s\n",
		path, task, modelID, definition.ID, reason)
	return nil
}

func executeCapability(
	repository, path string,
	task recipe.Task,
	capability capability,
	input string,
) error {
	ctx := context.Background()
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	inventory, err := capability.inventory(path)
	if err != nil {
		return err
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, inventory.Manifest.ID, task)
	if err != nil {
		return err
	}
	output, err := capability.execute(ctx, store, path, inventory.Manifest.ID, program, input)
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

func activate(
	repository, path, reason string,
	override sessionOverride,
	residency recipe.ResidencyPolicy,
	verification modelrecipe.Verification,
) error {
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
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
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
	modelPlan, err := model.CompileModelPlanWithProfile(spec, weights, profile)
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
		session, residency,
	)
	if err != nil {
		return err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceExperimental, reason); err != nil {
		return err
	}
	fmt.Printf("activated %s\n  model      %s\n  definition %s\n  recipe     %s\n  reason     %s\n",
		path, modelID, resolved.Document.ID, definition.ID, reason)
	return nil
}

func status(repository, path string, task recipe.Task) error {
	ctx := context.Background()
	var inventory modelartifact.Inventory
	if capability, ok := capabilities[task]; ok {
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
