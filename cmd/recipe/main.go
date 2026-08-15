// recipe: drives the model-recipe lifecycle for real artifacts — the command
// the sealed authority was waiting for (the lifecycle API was complete but
// uninvoked; the wave-1 probe recorded the refusal that proved it).
//
//	recipe verify -task <task> -input <json> <model>
//	                                        candidate execution evidence
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
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("recipe", run)
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: recipe <verify|activate|run|status> [options] <model>")
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
	input := flags.String("input", "", "task input as JSON; - reads standard input (verify, run)")
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
	case "verify":
		if !capabilityKnown || capability.execute == nil {
			return fmt.Errorf("task %q has no registered verifier runtime", selectedTask)
		}
		rawInput, inputErr := readInput(*input)
		if inputErr != nil {
			return inputErr
		}
		if strings.TrimSpace(rawInput) == "" {
			return errors.New("verify requires -input JSON")
		}
		return verifyCapability(repository, path, selectedTask, capability, rawInput)
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
		rawInput, inputErr := readInput(*input)
		if inputErr != nil {
			return inputErr
		}
		if strings.TrimSpace(rawInput) == "" {
			return errors.New("run requires -input JSON")
		}
		return executeCapability(repository, path, selectedTask, capability, rawInput)
	case "status":
		return status(repository, path, selectedTask)
	default:
		return fmt.Errorf("unknown verb %q", verb)
	}
}

func readInput(value string) (string, error) {
	if value != "-" {
		return value, nil
	}
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("recipe: read input: %w", err)
	}
	return string(data), nil
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
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, definition, err := prepareCapability(ctx, store, path, task, capability)
	if err != nil {
		return err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceExperimental, reason); err != nil {
		return err
	}
	fmt.Printf("activated %s\n  task       %s\n  model      %s\n  recipe     %s\n  reason     %s\n",
		path, task, modelID, definition.ID, reason)
	return nil
}

func prepareCapability(
	ctx context.Context,
	store artifact.Repository,
	path string,
	task recipe.Task,
	capability capability,
) (artifact.ID, recipe.Definition, error) {
	inventory, err := capability.inventory(path)
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	modelID := inventory.Manifest.ID
	var definition recipe.Definition
	var facts []artifact.Content
	if capability.bind != nil {
		definition, facts, err = capability.bind(path, modelID)
	} else {
		definition, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	batch, err := inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	batch.Contents = append(batch.Contents, facts...)
	if _, err := store.Commit(ctx, batch); err != nil {
		return artifact.ID{}, recipe.Definition{}, fmt.Errorf("publish model facts: %w", err)
	}
	return modelID, definition, nil
}

func verifyCapability(repository, path string, task recipe.Task, capability capability, input string) error {
	ctx := context.Background()
	revision, err := cleanGoRevision()
	if err != nil {
		return err
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, definition, err := prepareCapability(ctx, store, path, task, capability)
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
	started := time.Now()
	output, err := capability.execute(ctx, store, path, modelID, program, input)
	if err != nil {
		return err
	}
	verification, err := publishCapabilityVerification(ctx, store, definition, revision, time.Since(started))
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"gate_id": verification.Gate.String(), "output": output,
		"recipe_id": definition.ID.String(), "run_id": verification.Run.String(),
	})
}

func publishCapabilityVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
) (modelrecipe.Verification, error) {
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH,
		Device: "host", Backend: "go", Driver: "process", Runtime: runtime.Version(),
	})
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	duration := uint64(max(wall.Nanoseconds(), 1))
	record, err := runrecord.NewGateRecord(
		definition.ID, environment.ID, revision,
		runrecord.OutcomeSucceeded, "", duration,
		[]runrecord.GateStep{{
			Name: "candidate-execution", Phase: runrecord.PhaseTest,
			Outcome: runrecord.StepSucceeded, DurationNS: duration,
		}},
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch, err := record.Batch("recipe/verification/" + definition.ID.String() + "/" + record.Result.ID.String())
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch.Contents = append(batch.Contents, environmentContent)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, err
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

func cleanGoRevision() (string, error) {
	status, err := exec.Command("git", "status", "--porcelain", "--untracked-files=all", "--", "*.go").Output()
	if err != nil {
		return "", fmt.Errorf("recipe verifier source status: %w", err)
	}
	if strings.TrimSpace(string(status)) != "" {
		return "", errors.New("recipe verifier requires committed Go source")
	}
	revision, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("recipe verifier revision: %w", err)
	}
	return strings.TrimSpace(string(revision)), nil
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
	profile := spec.Profile()
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
