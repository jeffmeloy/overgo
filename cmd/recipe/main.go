// recipe: model-recipe lifecycle CLI.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

func main() {
	clioptions.MainNamed("recipe", run)
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: recipe <register|verify|activate|retire|retire-orphan|run|status|policy> [options] <model>")
	}
	verb := os.Args[1]
	if verb == "register" {
		return registerModels(os.Args[2:])
	}
	if verb == "validate-candidate" {
		return validateCandidate(os.Args[2:])
	}
	if verb == "attribute" {
		return attributeEvaluation(os.Args[2:])
	}
	if verb == "rollback" {
		return rollbackActivation(os.Args[2:])
	}
	if verb == "route" {
		return routeDecision(os.Args[2:])
	}
	if verb == "select" {
		return selectServing(os.Args[2:])
	}
	if verb == "replay" {
		return replayRouting(os.Args[2:])
	}
	if verb == "rollout-plan" {
		return declareRolloutPlan(os.Args[2:])
	}
	if verb == "assign" {
		return assignRollout(os.Args[2:])
	}
	if verb == "rollout-project" {
		return projectRollout(os.Args[2:])
	}
	if verb == "promote-rollout" {
		return promoteRollout(os.Args[2:])
	}
	if verb == "safety-window" {
		return deriveSafetyWindow(os.Args[2:])
	}
	if verb == "breaker" {
		return judgeBreaker(os.Args[2:])
	}
	if verb == "live-rollback" {
		return liveRollback(os.Args[2:])
	}
	if verb == "reenter" {
		return reenterQuarantine(os.Args[2:])
	}
	flags := flag.NewFlagSet("recipe "+verb, flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	reason := flags.String("reason", "", "activation reason recorded in the decision event (activate)")
	gate := flags.String("gate", "", "successful verifier gate artifact ID (activate)")
	runID := flags.String("run-id", "", "bound verifier run artifact ID (activate)")
	task := flags.String("task", string(recipe.TaskInference), "recipe task (inference|projection|forecast|tabular|seq2seq|speech|image-gen|video-gen|vqa)")
	projectorPath := flags.String("projector", "", "projector GGUF for projection recipes")
	sessionFlag := flags.String("session", "auto", "decode session: auto (derive from plan) | request | capacity")
	residencyFlag := flags.String("residency", string(recipe.ResidencyHybridNative),
		"weight residency: stream | host-cache | device-f32 | device-native | device-native-bf16 | hybrid-native | host-reference")
	input := flags.String("input", "", "task input as JSON; - reads standard input (verify, run)")
	alias := flags.String("alias", "", "OvergoDB model alias (run)")
	selection := flags.String("selection", string(modelrecipe.SessionWarm), "alias session selection: pin | warm | spillover (run)")
	compatibility := flags.String("compatibility", "", "remote peer compatibility evidence ID (spillover run)")
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
	capability, capabilityKnown := mediacapability.Catalog[selectedTask]
	if selectedTask == recipe.TaskProjection && verb != "status" {
		if strings.TrimSpace(*projectorPath) == "" {
			return errors.New("projection requires -projector")
		}
		configurationStore, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			return err
		}
		defer configurationStore.Close()
		capability = mediacapability.Projection(context.Background(), configurationStore, roots.ResolveModelPath(*projectorPath))
		capabilityKnown = true
	}
	switch verb {
	case "verify":
		if selectedTask == recipe.TaskInference || selectedTask == recipe.TaskProjection {
			rawInput, inputErr := readInput(*input)
			if inputErr != nil {
				return inputErr
			}
			if strings.TrimSpace(rawInput) == "" {
				return errors.New("verify requires -input JSON")
			}
			if selectedTask == recipe.TaskProjection {
				return verifyProjection(repository, path, roots.ResolveModelPath(*projectorPath), rawInput)
			}
			sessionOverride, sessionErr := parseSessionOverride(*sessionFlag)
			if sessionErr != nil {
				return sessionErr
			}
			residency, residencyErr := parseResidency(*residencyFlag)
			if residencyErr != nil {
				return residencyErr
			}
			return verifyInference(repository, path, rawInput, sessionOverride, residency)
		}
		if !capabilityKnown {
			return fmt.Errorf("task %q has no registered verifier runtime", selectedTask)
		}
		rawInput := ""
		if capability.Execute != nil {
			var inputErr error
			rawInput, inputErr = readInput(*input)
			if inputErr != nil {
				return inputErr
			}
			if strings.TrimSpace(rawInput) == "" {
				return errors.New("verify requires -input JSON")
			}
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
	case "retire":
		if selectedTask != recipe.TaskInference {
			return fmt.Errorf("retirement is unsupported for task %q", selectedTask)
		}
		if strings.TrimSpace(*reason) == "" {
			return errors.New("retire requires -reason: the refusal records why")
		}
		verification, verificationErr := parseVerification(*gate, *runID)
		if verificationErr != nil {
			return verificationErr
		}
		sessionOverride, sessionErr := parseSessionOverride(*sessionFlag)
		if sessionErr != nil {
			return sessionErr
		}
		residency, residencyErr := parseResidency(*residencyFlag)
		if residencyErr != nil {
			return residencyErr
		}
		return retire(repository, path, *reason, sessionOverride, residency, verification)
	case "run":
		if !capabilityKnown || capability.Execute == nil {
			return fmt.Errorf("task %q has no registered runtime", selectedTask)
		}
		rawInput, inputErr := readInput(*input)
		if inputErr != nil {
			return inputErr
		}
		if strings.TrimSpace(rawInput) == "" {
			return errors.New("run requires -input JSON")
		}
		selectedSession := modelrecipe.SessionSelection(*selection)
		if !selectedSession.Valid() {
			return fmt.Errorf("invalid session selection %q", *selection)
		}
		return executeCapability(
			repository, path, strings.TrimSpace(*alias), strings.TrimSpace(*compatibility),
			selectedSession, selectedTask, capability, rawInput,
		)
	case "status":
		return status(repository, path, selectedTask)
	case "policy":
		return ensurePolicy(repository, path, selectedTask)
	case "retire-orphan":
		if strings.TrimSpace(*reason) == "" {
			return errors.New("retire-orphan requires -reason: the refusal records why")
		}
		return retireOrphan(repository, flags.Arg(0), selectedTask, *reason)
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
	capability mediacapability.Capability,
	verification modelrecipe.Verification,
) error {
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, definition, err := prepareCapability(ctx, store, path, task, capability)
	if err != nil {
		return err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, reason); err != nil {
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
	capability mediacapability.Capability,
) (artifact.ID, recipe.Definition, error) {
	source, err := capability.Resolve(path)
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	modelID := source.Inventory.Manifest.ID
	var definition recipe.Definition
	var facts []artifact.Content
	if source.Define != nil {
		definition, facts, err = source.Define(modelID)
	} else {
		definition, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	batch, err := source.Inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	for _, related := range source.Related {
		relatedBatch, err := related.Batch(batch.Key)
		if err != nil {
			return artifact.ID{}, recipe.Definition{}, err
		}
		batch.Artifacts = append(batch.Artifacts, relatedBatch.Artifacts...)
		batch.Contents = append(batch.Contents, relatedBatch.Contents...)
		batch.Manifests = append(batch.Manifests, relatedBatch.Manifests...)
		batch.Lineage = append(batch.Lineage, relatedBatch.Lineage...)
		batch.Locations = append(batch.Locations, relatedBatch.Locations...)
	}
	batch.Contents = append(batch.Contents, facts...)
	if _, err := store.Commit(ctx, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return artifact.ID{}, recipe.Definition{}, fmt.Errorf("publish model facts: %w", err)
	}
	// Component-group manifests commit under their own content-derived
	// keys: the facts batch key is content-bound and predates them.
	for _, manifest := range source.Manifests {
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "recipe/component/" + manifest.ID.String(), Manifests: []artifact.Manifest{manifest},
		}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			return artifact.ID{}, recipe.Definition{}, fmt.Errorf("publish component manifest: %w", err)
		}
	}
	return modelID, definition, nil
}

func verifyCapability(repository, path string, task recipe.Task, capability mediacapability.Capability, input string) error {
	if capability.Execute == nil {
		return fmt.Errorf("task %q has no verifier executor", task)
	}
	ctx := context.Background()
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	_, definition, err := prepareCapability(ctx, store, path, task, capability)
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
	execution, err := modelrecipe.CompileCandidateExecution(ctx, store, program)
	if err != nil {
		return err
	}
	started := time.Now()
	var measured capabilityruntime.Measured
	output, err := capability.Execute(ctx, store, path, execution, input)
	if err != nil {
		return err
	}
	if envelope, ok := output.(capabilityruntime.Measured); ok {
		measured = envelope
	}
	verification, err := modelintake.PublishMeasuredVerification(
		ctx, store, definition, revision, time.Since(started), "host", "go", "candidate output validated", measured,
	)
	if err != nil {
		return err
	}
	result := map[string]any{
		"gate_id":   verification.Gate.String(),
		"recipe_id": definition.ID.String(), "run_id": verification.Run.String(),
		"output": output,
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}

func executeCapability(
	repository, path, alias, compatibilityText string,
	selection modelrecipe.SessionSelection,
	task recipe.Task,
	capability mediacapability.Capability,
	input string,
) error {
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	var compatibility artifact.ID
	if compatibilityText != "" {
		compatibility, err = artifact.ParseID(compatibilityText)
		if err != nil {
			return err
		}
	}
	var selected modelrecipe.CapabilityEvidenceSelection
	if alias != "" {
		selected, err = modelrecipe.ResolveCapabilityEvidenceSelector(
			ctx, store, modelrecipe.CapabilityEvidenceSelector{
				Alias: alias, Task: task, Session: selection, Compatibility: compatibility,
			},
		)
		if err != nil {
			return err
		}
		if selection == modelrecipe.SessionSpillover {
			// Spillover reaches the peer only through its exact derived
			// UTCP manual, and every invocation leaves a receipt chain
			// under the request's own operation identity.
			manual, manualErr := capabilityruntime.PeerCapabilityManual(selected, task)
			if manualErr != nil {
				return manualErr
			}
			operation, operationErr := artifact.IdentifyBytes(
				artifact.KindEvidence, []byte("overgo/peer-invocation/"+selected.Peer.ID.String()+"/"+string(task)),
			)
			if operationErr != nil {
				return operationErr
			}
			_, err = capabilityruntime.InvokeRemotePeer(
				ctx, http.DefaultClient, store, manual, selected, task,
				operation, strings.NewReader(input), os.Stdout,
			)
			return err
		}
	} else if selection == modelrecipe.SessionSpillover || compatibility.Valid() {
		return errors.New("recipe: spillover requires an evidence-bound alias")
	}
	source, err := capability.Resolve(path)
	if err != nil {
		return err
	}
	modelID := source.Inventory.Manifest.ID
	if alias != "" {
		if selected.Activation.Definition.Model != modelID {
			return errors.New("recipe: capability alias differs from loaded model")
		}
	} else {
		selected, err = modelrecipe.ResolveActiveExecution(ctx, store, modelID, task, selection)
		if err != nil {
			return err
		}
	}
	output, err := capability.Execute(ctx, store, path, selected, input)
	if err != nil {
		return err
	}
	// The run verb serves the output itself; measured decompositions
	// belong to verification evidence, not the serving contract.
	return json.NewEncoder(os.Stdout).Encode(capabilityruntime.Unwrap(output))
}

func parseSessionOverride(text string) (modelintake.SessionOverride, error) {
	switch strings.ToLower(strings.TrimSpace(text)) {
	case "", "auto":
		return modelintake.SessionOverride{}, nil
	case "request":
		return modelintake.SessionOverride{Set: true, Value: modelrecipe.DecodeSessionRequest}, nil
	case "capacity":
		return modelintake.SessionOverride{Set: true, Value: modelrecipe.DecodeSessionCapacity}, nil
	default:
		return modelintake.SessionOverride{}, fmt.Errorf("unknown -session %q (auto|request|capacity)", text)
	}
}

func activate(
	repository, path, reason string,
	override modelintake.SessionOverride,
	residency recipe.ResidencyPolicy,
	verification modelrecipe.Verification,
) error {
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	modelID := candidate.Inventory.Manifest.ID
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, candidate.Inventory, candidate.Resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, candidate.Definition, verification, recipe.EvidenceVerified, reason,
	); err != nil {
		return err
	}
	fmt.Printf("activated %s\n  model      %s\n  definition %s\n  recipe     %s\n  reason     %s\n",
		path, modelID, candidate.Resolved.Document.ID, candidate.Definition.ID, reason)
	return nil
}

func retire(
	repository, path, reason string,
	override modelintake.SessionOverride,
	residency recipe.ResidencyPolicy,
	verification modelrecipe.Verification,
) error {
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	if err := modelrecipe.RetireActiveCapability(
		ctx, store, candidate.Definition, verification, reason,
	); err != nil {
		return err
	}
	fmt.Printf("retired %s\n  model  %s\n  reason %s\n", path, candidate.Inventory.Manifest.ID, reason)
	return nil
}

// ensurePolicy republishes the catalog runtime policy for a model's
// ACTIVE recipe -- the migration repair for activations that predate
// the runtime-policy schema, whose golden suites may no longer be
// local for a full reverify. It touches nothing but the policy alias:
// the recipe, its evidence, and its activation stay exactly as
// committed.
func ensurePolicy(repository, path string, task recipe.Task) error {
	ctx := context.Background()
	inventory, err := taskModelInventory(path, task)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, inventory.Manifest.ID, task)
	if err != nil {
		return err
	}
	if !active {
		return fmt.Errorf("recipe: %s has no active %s recipe to bind a policy to", path, task)
	}
	if err := modelrecipe.EnsureRuntimePolicy(ctx, store, activation.Definition); err != nil {
		return err
	}
	policy, err := modelrecipe.ResolveRuntimePolicy(ctx, store, activation.Definition)
	if err != nil {
		return err
	}
	fmt.Printf("runtime policy %s bound to recipe %s\n", policy.ID, activation.Definition.ID)
	return nil
}

// taskModelInventory shares task-specific identity resolution between status
// and policy operations through the existing capability source owners.
func taskModelInventory(path string, task recipe.Task) (modelartifact.Inventory, error) {
	if task == recipe.TaskVQA {
		return modelartifact.FromHFPath(path)
	}
	if capability, known := mediacapability.Catalog[task]; known && capability.Resolve != nil {
		source, err := capability.Resolve(path)
		if err != nil {
			return modelartifact.Inventory{}, err
		}
		return source.Inventory, nil
	}
	if task != recipe.TaskInference && task != recipe.TaskProjection {
		return modelartifact.Inventory{}, fmt.Errorf("unsupported model recipe task %q", task)
	}
	file, err := gguf.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, err
	}
	defer file.Close()
	return modelartifact.FromGGUF(file, artifact.KindModel)
}

func status(repository, path string, task recipe.Task) error {
	ctx := context.Background()
	inventory, err := taskModelInventory(path, task)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
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

// registerModels publishes exact physical inventories and their supplied source
// declaration. It never derives recipes, selects runtimes or changes activation.
func registerModels(args []string) error {
	flags := flag.NewFlagSet("recipe register", flag.ContinueOnError)
	repository := flags.String("repo", "", "OvergoDB store")
	root := flags.String("root", "", "root containing the declared model directories")
	declaration := flags.String("spec", "", "JSON array of exact model, tensor, source and license identities")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repository == "" || *root == "" || *declaration == "" {
		return errors.New("usage: recipe register -repo STORE -root MODELS -spec JSON")
	}
	ctx := context.Background()
	batch, err := modelartifact.PrepareRegistration(ctx, *root, *declaration)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if err := modelartifact.PreserveRawTextDescriptors(ctx, store, &batch); err != nil {
		return err
	}
	// Include physical locations in the idempotency key: the same exact models
	// may be registered at another root without changing their identities.
	identityBytes, err := json.Marshal(batch)
	if err != nil {
		return err
	}
	identity, err := artifact.IdentifyBytes(artifact.KindFile, identityBytes)
	if err != nil {
		return err
	}
	batch.Key = "recipe/registration/" + identity.DigestHex()
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	fmt.Printf("registered=%d source_declaration=%s; exact model, tensor, source and license identities checked; model bytes copied=0; runtime execution and activation did not run\n", len(batch.Manifests), batch.Contents[0].Descriptor.ID)
	return nil
}
