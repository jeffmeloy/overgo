package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingattempt"
	"overgo/internal/tokenizer"
)

// The failure codes a measured inference attempt records: its model did not
// load, or its request did not generate.
const (
	loadFailed       = "load_failed"
	generationFailed = "generation_failed"
)

// servedModel is what one measured request needs of a loaded model.
type servedModel interface {
	evaluation.Generator
	DeviceMemoryStats(context.Context) (driver.MemoryStats, error)
	Close() error
}

// measureRunner measures the model at path on the runner its active recipe
// resolves, as every serving command opens it, and prints the request's result.
func measureRunner(repository, path string) error {
	result, err := measureInference(context.Background(), repository, path, func(ctx context.Context) (servedModel, error) {
		runner, err := modelcli.OpenRunner(ctx, repository, path, inference.OpenOptions{})
		if err != nil {
			return nil, err
		}
		return runner, nil
	})
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}

// measureInference runs the active inference recipe of the model at path once,
// on the first request its verification ran, and records the run as the
// serving attempt the scorecard reads; open loads the model. The store is
// written only once the model is closed, so the writer never spans model
// execution.
func measureInference(ctx context.Context, repository, path string, open func(context.Context) (servedModel, error)) (evaluation.ExactResult, error) {
	inventory, err := taskModelInventory(path, recipe.TaskInference)
	if err != nil {
		return evaluation.ExactResult{}, err
	}
	reader, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return evaluation.ExactResult{}, err
	}
	active, request, err := verifiedCase(ctx, reader, inventory.Manifest.ID)
	if err = errors.Join(err, reader.Close()); err != nil {
		return evaluation.ExactResult{}, err
	}
	observation, result, cause := observeInference(ctx, open, request)
	observation.Model, observation.Recipe = inventory.Manifest.ID, active
	store, err := overgodb.Open(repository)
	if err != nil {
		return result, errors.Join(cause, err)
	}
	recorded := errors.Join(recordInference(context.WithoutCancel(ctx), store, observation), store.Close())
	return result, errors.Join(cause, recorded)
}

// verifiedCase finds the request an active inference recipe's verification
// ran first: the first case of the exact suite its activation's gate replayed,
// which a DNA model's verification states in its own alphabet.
func verifiedCase(ctx context.Context, store *overgodb.Store, model artifact.ID) (active artifact.ID, request evaluation.ExactCase, err error) {
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, model, recipe.TaskInference)
	if err != nil || !found {
		return active, request, cmp.Or(err, fmt.Errorf("recipe: %s has no active inference recipe", model))
	}
	verified := activation.Verified
	// A gate records at least one step, its candidate execution first.
	planID, _ := exactEvidence(verified.Gate.Steps[0].Evidence)
	content, err := artifact.RequireTypedContent(ctx, store, planID)
	if err != nil {
		return active, request, fmt.Errorf("recipe: the verification of %s names no exact plan; verify it again with -input: %w", model, err)
	}
	plan, err := evaluation.ParsePlan(content.Data)
	if err != nil {
		return active, request, err
	}
	dataset, err := artifact.RequireTypedContent(ctx, store, plan.Dataset())
	if err != nil {
		return active, request, err
	}
	var cases []evaluation.ExactCase
	if err := json.Unmarshal(dataset.Data, &cases); err != nil || len(cases) == 0 {
		return active, request, errors.Join(err, fmt.Errorf("recipe: the exact suite %s holds no case", plan.Dataset()))
	}
	return activation.Definition.ID, cases[0], nil
}

// observeInference loads the model and runs the request on it the way its
// verification ran it, and returns the attempt as a serving observation: the
// load, prefill and decode it timed, the tokens it gave, and device memory at
// the start, the prefill and the finish, timed from the request's start as the
// server times a page's. It also returns the request's result and the failure
// the attempt met: a model that did not load or a request that did not
// generate is a failed attempt. A span the clock could not time is left out.
func observeInference(ctx context.Context, open func(context.Context) (servedModel, error), request evaluation.ExactCase) (runrecord.ServingObservation, evaluation.ExactResult, error) {
	observation := runrecord.ServingObservation{
		Version: artifact.InitialDocumentVersion, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: time.Now().UnixNano(),
	}
	var walls processmeasure.Walls
	clock := processmeasure.NewStopwatch()
	model, err := open(ctx)
	load := walls.Elapsed(clock)
	generation := &observedGeneration{servedModel: model, clock: processmeasure.NewStopwatch()}
	var result evaluation.ExactResult
	var total uint64
	failure, released := loadFailed, error(nil)
	if err == nil {
		generation.sample(ctx, runrecord.ServingHardwareStart)
		result, err = evaluation.Record(ctx, generation, request.Name, request.Prompt, request.MaxTokens)
		generation.sample(ctx, runrecord.ServingHardwareFinish)
		total, failure, released = generation.walls.Elapsed(generation.clock), generationFailed, model.Close()
	}
	if err != nil {
		observation.Outcome, observation.Failure = runrecord.OutcomeFailed, failure
	}
	prefill := min(generation.prefill, total)
	observation.MeasuredNS, observation.Hardware = load+total, generation.samples
	observation.Usage.OutputTokens = uint64(max(result.GeneratedTokens, 0))
	observation.Phases = slices.DeleteFunc([]runrecord.PhaseMetric{
		{Phase: runrecord.PhaseLoad, DurationNS: load}, {Phase: runrecord.PhasePrefill, DurationNS: prefill}, {Phase: runrecord.PhaseDecode, DurationNS: total - prefill},
	}, func(phase runrecord.PhaseMetric) bool { return phase.DurationNS == 0 })
	// A model that served its request but did not release is no failed
	// attempt; its release failure is still the command's.
	return observation, result, errors.Join(err, released)
}

// observedGeneration is a loaded model whose generation reports what the
// server's per-request observation reads: the prefill it waited for, and
// device memory sampled at the stages of the request.
type observedGeneration struct {
	servedModel
	clock   processmeasure.Stopwatch
	walls   processmeasure.Walls
	prefill uint64
	samples []runrecord.ServingHardwareSample
}

// Generate runs the model, timing its prefill and sampling device memory when
// the prompt is evaluated.
func (generation *observedGeneration) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	evaluated := options.OnPromptEvaluated
	options.OnPromptEvaluated = func(prompted inference.PromptEvaluation) {
		generation.prefill += uint64(max(prompted.Duration, 0))
		generation.sample(ctx, runrecord.ServingHardwarePrefill)
		if evaluated != nil {
			evaluated(prompted)
		}
	}
	return generation.servedModel.Generate(ctx, prompt, options)
}

// sample records device memory at one stage of the request, the prefill once.
// A model whose device memory cannot be read records none, as the server's
// does.
func (generation *observedGeneration) sample(ctx context.Context, stage runrecord.ServingHardwareStage) {
	if slices.ContainsFunc(generation.samples, func(sample runrecord.ServingHardwareSample) bool { return sample.Stage == stage }) {
		return
	}
	stats, err := generation.DeviceMemoryStats(ctx)
	if err != nil {
		return
	}
	generation.samples = append(generation.samples, runrecord.ServingHardwareSample{
		Stage: stage, ElapsedNS: generation.walls.Elapsed(generation.clock),
		DeviceCurrentBytes: stats.CurrentBytes, DevicePeakBytes: stats.PeakBytes, DeviceAllocations: stats.Allocations,
	})
}

// recordInference publishes the attempt under the environment it ran in, the
// one the server's attempts run under, through the owner that also parks a
// failed attempt as a finding.
func recordInference(ctx context.Context, store artifact.Repository, observation runrecord.ServingObservation) error {
	environment, err := runrecord.CurrentEnvironment("process", "go")
	if err != nil {
		return err
	}
	batch, err := environment.Batch("serving/environment/" + environment.ID.String())
	if err != nil {
		return err
	}
	// The environment of an earlier attempt on this machine is already the
	// stored fact.
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return err
	}
	observation.Environment = environment.ID
	_, err = servingattempt.Record(ctx, store, observation)
	return err
}
