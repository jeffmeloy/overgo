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
	"os/exec"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("recipe", run)
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: recipe <verify|activate|retire|retire-orphan|run|status|policy> [options] <model>")
	}
	verb := os.Args[1]
	if verb == "declare-prototype" {
		return declarePrototype(os.Args[2:])
	}
	if verb == "compile-trial" {
		return compilePrototypeTrial(os.Args[2:])
	}
	if verb == "materialize-trial" {
		return materializeTrial(os.Args[2:])
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
	capability, capabilityKnown := capabilities[selectedTask]
	if selectedTask == recipe.TaskProjection && verb != "status" {
		if strings.TrimSpace(*projectorPath) == "" {
			return errors.New("projection requires -projector")
		}
		capability = projectionCapability(roots.ResolveModelPath(*projectorPath))
		capabilityKnown = true
	}
	switch verb {
	case "verify":
		if selectedTask == recipe.TaskInference {
			rawInput, inputErr := readInput(*input)
			if inputErr != nil {
				return inputErr
			}
			if strings.TrimSpace(rawInput) == "" {
				return errors.New("verify requires -input JSON")
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
		if capability.execute != nil {
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
	capability capability,
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
	capability capability,
) (artifact.ID, recipe.Definition, error) {
	source, err := capability.resolve(path)
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	modelID := source.inventory.Manifest.ID
	var definition recipe.Definition
	var facts []artifact.Content
	if source.define != nil {
		definition, facts, err = source.define(modelID)
	} else {
		definition, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	batch, err := source.inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	for _, related := range source.related {
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
	if _, err := store.Commit(ctx, batch); err != nil {
		return artifact.ID{}, recipe.Definition{}, fmt.Errorf("publish model facts: %w", err)
	}
	// Component-group manifests commit under their own content-derived
	// keys: the facts batch key is content-bound and predates them.
	for _, manifest := range source.manifests {
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "recipe/component/" + manifest.ID.String(), Manifests: []artifact.Manifest{manifest},
		}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			return artifact.ID{}, recipe.Definition{}, fmt.Errorf("publish component manifest: %w", err)
		}
	}
	return modelID, definition, nil
}

func verifyCapability(repository, path string, task recipe.Task, capability capability, input string) error {
	ctx := context.Background()
	revision, err := cleanGoRevision()
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
	var output any
	var measured capabilityruntime.Measured
	if capability.execute != nil {
		output, err = capability.execute(ctx, store, path, execution, input)
		if err != nil {
			return err
		}
		if envelope, ok := output.(capabilityruntime.Measured); ok {
			measured = envelope
		}
	}
	verification, err := publishMeasuredVerification(
		ctx, store, definition, revision, time.Since(started), "host", "go", "candidate output validated", measured,
	)
	if err != nil {
		return err
	}
	result := map[string]any{
		"gate_id":   verification.Gate.String(),
		"recipe_id": definition.ID.String(), "run_id": verification.Run.String(),
	}
	if capability.execute != nil {
		result["output"] = output
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}

func publishCapabilityVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence string,
) (modelrecipe.Verification, error) {
	return publishMeasuredVerification(
		ctx, store, definition, revision, wall, device, backend, evidence, capabilityruntime.Measured{},
	)
}

// publishMeasuredVerification records a successful candidate execution
// together with its measured decomposition: per-node phase walls become
// gate steps (and thereby run phases), and the peak device bytes enter
// the execution step's evidence.
func publishMeasuredVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence string,
	measured capabilityruntime.Measured,
) (modelrecipe.Verification, error) {
	return publishCapabilityResult(
		ctx, store, definition, revision, wall, device, backend, evidence,
		runrecord.OutcomeSucceeded, "", measured,
	)
}

func publishCapabilityFailure(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence, failure string,
) (modelrecipe.Verification, error) {
	return publishCapabilityResult(
		ctx, store, definition, revision, wall, device, backend, evidence,
		runrecord.OutcomeFailed, failure, capabilityruntime.Measured{},
	)
}

// mediaNodePhase maps recipe node identifiers onto the run phase
// vocabulary; nodes outside the vocabulary carry no phase step and
// remain inside the total wall.
func mediaNodePhase(node recipe.NodeID) (runrecord.Phase, bool) {
	switch node {
	case "prepare":
		return runrecord.PhasePrepare, true
	case "integrate":
		return runrecord.PhaseIntegrate, true
	case "decode":
		return runrecord.PhaseDecode, true
	case "generate":
		return runrecord.PhaseGenerate, true
	case "tokenize":
		return runrecord.PhaseTokenize, true
	}
	return "", false
}

func publishCapabilityResult(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence string,
	outcome runrecord.Outcome,
	failure string,
	measured capabilityruntime.Measured,
) (modelrecipe.Verification, error) {
	environment, err := runrecord.CurrentEnvironment(device, backend)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	duration := uint64(max(wall.Nanoseconds(), 1))
	stepOutcome := runrecord.StepSucceeded
	if outcome == runrecord.OutcomeFailed {
		stepOutcome = runrecord.StepFailed
	}
	if measured.PeakDeviceBytes > 0 {
		evidence = fmt.Sprintf("%s; peak_device_bytes=%d", evidence, measured.PeakDeviceBytes)
	}
	steps := []runrecord.GateStep{{
		Name: "candidate-execution", Phase: runrecord.PhaseTest,
		Outcome: stepOutcome, DurationNS: duration, Evidence: evidence,
	}}
	for _, node := range measured.Phases {
		phase, ok := mediaNodePhase(node.Node)
		if !ok || node.WallNS == 0 {
			continue
		}
		steps = append(steps, runrecord.GateStep{
			Name: "node-" + string(node.Node), Phase: phase,
			Outcome: stepOutcome, DurationNS: node.WallNS,
		})
	}
	record, err := runrecord.NewGateRecord(
		definition.ID, environment.ID, revision,
		outcome, failure, duration, steps,
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
	return runrecord.HeadCommit(".")
}

func executeCapability(
	repository, path, alias, compatibilityText string,
	selection modelrecipe.SessionSelection,
	task recipe.Task,
	capability capability,
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
			return capabilityruntime.ExecuteRemotePeer(ctx, http.DefaultClient, selected, strings.NewReader(input), os.Stdout)
		}
	} else if selection == modelrecipe.SessionSpillover || compatibility.Valid() {
		return errors.New("recipe: spillover requires an evidence-bound alias")
	}
	source, err := capability.resolve(path)
	if err != nil {
		return err
	}
	modelID := source.inventory.Manifest.ID
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
	output, err := capability.execute(ctx, store, path, selected, input)
	if err != nil {
		return err
	}
	// The run verb serves the output itself; measured decompositions
	// belong to verification evidence, not the serving contract.
	return json.NewEncoder(os.Stdout).Encode(capabilityruntime.Unwrap(output))
}

// sessionOverride: operator-pinned decode session; nil defers to the
// plan-derived choice.
type sessionOverride struct {
	set   bool
	value modelrecipe.DecodeSessionPolicy
}

type inferenceCandidate struct {
	inventory  modelartifact.Inventory
	resolved   modelrecipe.ResolvedModelDefinition
	definition recipe.Definition
}

func prepareInferenceCandidate(
	ctx context.Context,
	store artifact.Reader,
	path string,
	override sessionOverride,
	residency recipe.ResidencyPolicy,
) (inferenceCandidate, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return inferenceCandidate{}, err
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		return inferenceCandidate{}, err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return inferenceCandidate{}, err
	}
	profileDocument, err := modelrecipe.ResolveRegisteredArchitectureProfile(ctx, store, spec.Architecture)
	if err != nil {
		return inferenceCandidate{}, err
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	if err != nil {
		return inferenceCandidate{}, err
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return inferenceCandidate{}, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return inferenceCandidate{}, err
	}
	modelPlan, err := model.CompileModelPlanWithProfile(spec, weights, profileDocument.Policy)
	if err != nil {
		return inferenceCandidate{}, err
	}
	session := modelrecipe.DecodeSessionCapacity
	if !modelPlan.SupportsCapacityCache() {
		session = modelrecipe.DecodeSessionRequest
	}
	if override.set {
		if override.value == modelrecipe.DecodeSessionCapacity && !modelPlan.SupportsCapacityCache() {
			return inferenceCandidate{}, errors.New("recipe: -session capacity is unsupported by this model's compiled plan")
		}
		session = override.value
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		session, residency,
	)
	if err != nil {
		return inferenceCandidate{}, err
	}
	return inferenceCandidate{inventory: inventory, resolved: resolved, definition: definition}, nil
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
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := prepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	modelID := candidate.inventory.Manifest.ID
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, candidate.inventory, candidate.resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, candidate.definition, verification, recipe.EvidenceVerified, reason,
	); err != nil {
		return err
	}
	fmt.Printf("activated %s\n  model      %s\n  definition %s\n  recipe     %s\n  reason     %s\n",
		path, modelID, candidate.resolved.Document.ID, candidate.definition.ID, reason)
	return nil
}

func retire(
	repository, path, reason string,
	override sessionOverride,
	residency recipe.ResidencyPolicy,
	verification modelrecipe.Verification,
) error {
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := prepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	if err := modelrecipe.RetireActiveCapability(
		ctx, store, candidate.definition, verification, reason,
	); err != nil {
		return err
	}
	fmt.Printf("retired %s\n  model  %s\n  reason %s\n", path, candidate.inventory.Manifest.ID, reason)
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
	inventory, err := policyModelInventory(path, task)
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

// policyModelInventory resolves the model identity for a policy bind
// through the same authority each task uses everywhere else: the GGUF
// header for inference-family models, the task's registered capability
// source for everything else -- a safetensors speech or media model is
// never opened as a GGUF.
func policyModelInventory(path string, task recipe.Task) (modelartifact.Inventory, error) {
	if task == recipe.TaskVQA {
		return modelartifact.FromHFPath(path)
	}
	if capability, known := capabilities[task]; known && capability.resolve != nil {
		source, err := capability.resolve(path)
		if err != nil {
			return modelartifact.Inventory{}, err
		}
		return source.inventory, nil
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
	var inventory modelartifact.Inventory
	if task == recipe.TaskVQA {
		var err error
		inventory, err = modelartifact.FromHFPath(path)
		if err != nil {
			return err
		}
	} else if capability, ok := capabilities[task]; ok {
		source, err := capability.resolve(path)
		if err != nil {
			return err
		}
		inventory = source.inventory
	} else {
		if task != recipe.TaskInference && task != recipe.TaskProjection {
			return fmt.Errorf("unsupported model recipe task %q", task)
		}
		file, err := gguf.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		inventory, err = modelartifact.FromGGUF(file, artifact.KindModel)
		if err != nil {
			return err
		}
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
