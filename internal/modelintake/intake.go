//overgo:runtime-inputs caller

// Package modelintake owns the steps that take a model file into the
// store's catalog: preparing its inference candidate (the model's facts,
// definition and recipe), registering that candidate, recording and
// replaying an exact suite, and publishing the verification evidence an
// activation binds. cmd/recipe sequences these for the command line and
// the server sequences them as operations behind the GUI's library
// (professional GUI campaign, gui-library); neither carries the logic.
package modelintake

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// SessionOverride is an operator-pinned decode session; unset defers to the
// compiled plan (capacity when it supports the capacity cache).
type SessionOverride struct {
	Set   bool
	Value modelrecipe.DecodeSessionPolicy
}

// Candidate is one model file resolved against its registered architecture
// profile: the inventory of its tensors, the resolved definition and the
// inference recipe definition that would serve it.
type Candidate struct {
	Inventory  modelartifact.Inventory
	Resolved   modelrecipe.ResolvedModelDefinition
	Definition recipe.Definition
}

// PrepareInferenceCandidate reads a GGUF, resolves it against the store's
// registered architecture profile and compiles its plan to choose the
// decode session, returning the candidate without publishing anything.
func PrepareInferenceCandidate(
	ctx context.Context,
	store artifact.Reader,
	path string,
	override SessionOverride,
	residency recipe.ResidencyPolicy,
) (Candidate, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return Candidate{}, err
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		return Candidate{}, err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return Candidate{}, err
	}
	profileDocument, err := modelrecipe.ResolveRegisteredArchitectureProfile(ctx, store, spec.Architecture)
	if err != nil {
		return Candidate{}, err
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	if err != nil {
		return Candidate{}, err
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return Candidate{}, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return Candidate{}, err
	}
	modelPlan, err := model.CompileModelPlanWithProfile(spec, weights, profileDocument.Policy)
	if err != nil {
		return Candidate{}, err
	}
	session := modelrecipe.DecodeSessionCapacity
	if !modelPlan.SupportsCapacityCache() {
		session = modelrecipe.DecodeSessionRequest
	}
	if override.Set {
		if override.Value == modelrecipe.DecodeSessionCapacity && !modelPlan.SupportsCapacityCache() {
			return Candidate{}, errors.New("model intake: the capacity session is unsupported by this model's compiled plan")
		}
		session = override.Value
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		session, residency,
	)
	if err != nil {
		return Candidate{}, err
	}
	return Candidate{Inventory: inventory, Resolved: resolved, Definition: definition}, nil
}

// RegisterCandidate publishes the model's resolved facts and, when the
// candidate recipe is not yet published, the candidate itself: the model
// is then known to the store as a candidate, not yet servable.
func RegisterCandidate(ctx context.Context, store artifact.Repository, candidate Candidate) error {
	// Facts already published are no change, and a registration repeated
	// for a model the store knows is a registration, not a fault.
	if _, err := modelrecipe.PublishResolvedModelDefinition(ctx, store, candidate.Inventory, candidate.Resolved); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return fmt.Errorf("publish model facts: %w", err)
	}
	_, published, err := modelrecipe.Status(ctx, store, candidate.Definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+candidate.Definition.ID.String(), candidate.Definition,
		); err != nil {
			return err
		}
	}
	return nil
}

// RecordExactSuite generates each prompt greedily and records what the
// model produced as the case's expected text: a fresh model's first
// golden, which a replay then has to reproduce exactly.
func RecordExactSuite(ctx context.Context, generator evaluation.Generator, source string, prompts []string, maxTokens int) (evaluation.ExactSuite, error) {
	if generator == nil || strings.TrimSpace(source) == "" || len(prompts) == 0 || maxTokens <= 0 {
		return evaluation.ExactSuite{}, errors.New("model intake: a generator, a source, prompts and a token bound are required")
	}
	suite := evaluation.ExactSuite{Schema: "overgo/exact-suite/recorded", Source: source, Date: time.Now().UTC().Format(time.DateOnly)}
	for index, prompt := range prompts {
		name := "recorded-" + strconv.Itoa(index)
		result, err := evaluation.Record(ctx, generator, name, prompt, maxTokens)
		if err != nil {
			return evaluation.ExactSuite{}, err
		}
		if result.GeneratedTokens <= 0 || result.PromptTokens <= 0 {
			return evaluation.ExactSuite{}, fmt.Errorf("model intake: case %q generated nothing", name)
		}
		suite.Cases = append(suite.Cases, evaluation.ExactCase{
			Name: name, Prompt: prompt, MaxTokens: maxTokens, Text: result.Text,
			PromptTokens: result.PromptTokens, GeneratedTokens: result.GeneratedTokens,
		})
	}
	return suite, nil
}

// PublishVerification records a successful candidate execution as gate and
// run evidence bound to the recipe, the code revision and the environment.
func PublishVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence string,
) (modelrecipe.Verification, error) {
	return PublishMeasuredVerification(ctx, store, definition, revision, wall, device, backend, evidence, capabilityruntime.Measured{})
}

// PublishMeasuredVerification records a successful candidate execution
// together with its measured decomposition: per-node phase walls become
// gate steps (and thereby run phases), and the peak device bytes enter
// the execution step's evidence. A supplied request becomes a retained run
// input; absent requests remain unbound.
func PublishMeasuredVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence string,
	measured capabilityruntime.Measured,
) (modelrecipe.Verification, error) {
	return publishResult(ctx, store, definition, revision, wall, device, backend, evidence, runrecord.OutcomeSucceeded, "", measured)
}

// PublishFailure records a failed candidate execution with its failure.
func PublishFailure(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	revision string,
	wall time.Duration,
	device, backend, evidence, failure string,
) (modelrecipe.Verification, error) {
	return publishResult(ctx, store, definition, revision, wall, device, backend, evidence, runrecord.OutcomeFailed, failure, capabilityruntime.Measured{})
}

// nodePhase maps recipe node identifiers onto the run phase vocabulary;
// nodes outside the vocabulary carry no phase step and remain inside the
// total wall.
func nodePhase(node recipe.NodeID) (runrecord.Phase, bool) {
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

func publishResult(
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
	var inputs []artifact.ID
	if measured.Input.Descriptor != (artifact.Descriptor{}) || measured.Input.Data != nil {
		if err := measured.Input.Validate(); err != nil {
			return modelrecipe.Verification{}, fmt.Errorf("capability verification request: %w", err)
		}
		inputs = []artifact.ID{measured.Input.Descriptor.ID}
	}
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
		phase, ok := nodePhase(node.Node)
		if !ok || node.WallNS == 0 {
			continue
		}
		steps = append(steps, runrecord.GateStep{
			Name: "node-" + string(node.Node), Phase: phase,
			Outcome: stepOutcome, DurationNS: node.WallNS,
		})
	}
	record, err := runrecord.NewGateRecord(definition.ID, environment.ID, revision, outcome, failure, duration, steps)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	publicationID := record.Result.ID
	if len(inputs) != 0 {
		run := record.Run
		record.Run, err = runrecord.NewBoundRun(run.Recipe, run.Outcome, inputs, run.Outputs, run.Failure, run.CodeCommit, run.Environment, run.MeasuredNS, run.Phases)
		if err != nil {
			return modelrecipe.Verification{}, err
		}
		publicationID = record.Run.ID
	}
	batch, err := record.Batch("recipe/verification/" + definition.ID.String() + "/" + publicationID.String())
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if len(inputs) != 0 {
		batch.Contents = append(batch.Contents, measured.Input)
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

// CleanRevision names the committed Go source a verification binds to; a
// dirty Go tree has no revision to bind and is refused.
func CleanRevision(ctx context.Context) (string, error) {
	var status bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: []string{"status", "--porcelain", "--untracked-files=all", "--", "*.go"}, Stdout: &status,
	})
	if err != nil || receipt.ExitCode != 0 {
		return "", fmt.Errorf("model intake: verifier source status: %w", errors.Join(err, fmt.Errorf("git exited %d", receipt.ExitCode)))
	}
	if strings.TrimSpace(status.String()) != "" {
		return "", errors.New("model intake: a verification requires committed Go source")
	}
	return runrecord.HeadCommit(".")
}
