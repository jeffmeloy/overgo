package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
// Asked for the journey, it prints where a first request's time goes and
// records nothing (journeyInference).
func measureRunner(repository, path string, journey bool) error {
	open := func(ctx context.Context) (servedModel, error) {
		runner, err := modelcli.OpenRunner(ctx, repository, path, inference.OpenOptions{})
		if err != nil {
			return nil, err
		}
		return runner, nil
	}
	if journey {
		return journeyInference(context.Background(), repository, path, open, os.Stdout)
	}
	result, err := measureInference(context.Background(), repository, path, open)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}

// measuredRequest finds, for the model at path, its identity, its active
// inference recipe and the requests that recipe's verification ran, in its
// order and at least one. A measurement runs the first.
func measuredRequest(ctx context.Context, repository, path string) (model, active artifact.ID, requests []evaluation.ExactCase, err error) {
	inventory, err := taskModelInventory(path, recipe.TaskInference)
	if err != nil {
		return model, active, requests, err
	}
	reader, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return model, active, requests, err
	}
	active, requests, err = verifiedCase(ctx, reader, inventory.Manifest.ID)
	return inventory.Manifest.ID, active, requests, errors.Join(err, reader.Close())
}

// journeyInference prints where the time of a first request goes on the model
// at path, several ways, and records nothing: a first load in this process
// and the measured request on it, a second load in this process, the same
// request again on a model that has served it, and, where the recipe's
// verification ran a second request, that other request on a model that has
// served the first. Each way prints the parts of its load that name
// themselves, and how long after the request began its prompt was evaluated
// and its first token came. The load of a way that serves a request first
// holds that request, so only the request line of such a way is a reading.
func journeyInference(ctx context.Context, repository, path string, open func(context.Context) (servedModel, error), said io.Writer) error {
	_, _, requests, err := measuredRequest(ctx, repository, path)
	if err != nil {
		return err
	}
	request := requests[0]
	served := func(ctx context.Context) (servedModel, error) {
		model, err := open(ctx)
		if err != nil {
			return nil, err
		}
		if _, err := evaluation.Record(ctx, model, request.Name, request.Prompt, request.MaxTokens); err != nil {
			return nil, errors.Join(err, model.Close())
		}
		return model, nil
	}
	type way struct {
		name    string
		open    func(context.Context) (servedModel, error)
		request evaluation.ExactCase
	}
	ways := []way{
		{"a first load in this process", open, request}, {"a second load in this process", open, request},
		{"the same request again on a loaded model", served, request},
	}
	if len(requests) > 1 {
		ways = append(ways, way{"another request on a loaded model", served, requests[1]})
	}
	for _, way := range ways {
		fmt.Fprintf(said, "recipe: %s\n", way.name)
		if _, _, err := observeInference(ctx, way.open, way.request, said); err != nil {
			return err
		}
	}
	return nil
}

// measureInference runs the active inference recipe of the model at path once,
// on the first request its verification ran, and records the run as the
// serving attempt the scorecard reads; open loads the model. The store is
// written only once the model is closed, so the writer never spans model
// execution. Where the load's time went, and when the first token came, goes
// beside the result on the other stream.
func measureInference(ctx context.Context, repository, path string, open func(context.Context) (servedModel, error)) (evaluation.ExactResult, error) {
	model, active, requests, err := measuredRequest(ctx, repository, path)
	if err != nil {
		return evaluation.ExactResult{}, err
	}
	observation, result, cause := observeInference(ctx, open, requests[0], os.Stderr)
	observation.Model, observation.Recipe = model, active
	store, err := overgodb.Open(repository)
	if err != nil {
		return result, errors.Join(cause, err)
	}
	recorded := errors.Join(recordInference(context.WithoutCancel(ctx), store, observation), store.Close())
	return result, errors.Join(cause, recorded)
}

// verifiedCase finds the requests an active inference recipe's verification
// ran: the cases of the exact suite its activation's gate replayed, in order,
// which a DNA model's verification states in its own alphabet.
func verifiedCase(ctx context.Context, store *overgodb.Store, model artifact.ID) (active artifact.ID, requests []evaluation.ExactCase, err error) {
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, model, recipe.TaskInference)
	if err != nil || !found {
		return active, nil, cmp.Or(err, fmt.Errorf("recipe: %s has no active inference recipe", model))
	}
	verified := activation.Verified
	// A gate records at least one step, its candidate execution first.
	planID, _ := exactEvidence(verified.Gate.Steps[0].Evidence)
	content, err := artifact.RequireTypedContent(ctx, store, planID)
	if err != nil {
		return active, nil, fmt.Errorf("recipe: the verification of %s names no exact plan; verify it again with -input: %w", model, err)
	}
	plan, err := evaluation.ParsePlan(content.Data)
	if err != nil {
		return active, nil, err
	}
	dataset, err := artifact.RequireTypedContent(ctx, store, plan.Dataset())
	if err != nil {
		return active, nil, err
	}
	var cases []evaluation.ExactCase
	if err := json.Unmarshal(dataset.Data, &cases); err != nil || len(cases) == 0 {
		return active, nil, errors.Join(err, fmt.Errorf("recipe: the exact suite %s holds no case", plan.Dataset()))
	}
	return activation.Definition.ID, cases, nil
}

// observeInference loads the model and runs the request on it the way its
// verification ran it, and returns the attempt as a serving observation: the
// load, prefill and decode it timed, the tokens it gave, and device memory at
// the start, the prefill and the finish, timed from the request's start as the
// server times a page's. It also returns the request's result and the failure
// the attempt met: a model that did not load or a request that did not
// generate is a failed attempt. A span the clock could not time is left out.
//
// It writes to said where the load's time went, by the parts of the load that
// name themselves, and when the request's prompt was evaluated, when device
// memory was sampled for it, which is the time the scorecard reads as first
// output, and when its first token came, each from the request's start.
func observeInference(ctx context.Context, open func(context.Context) (servedModel, error), request evaluation.ExactCase, said io.Writer) (runrecord.ServingObservation, evaluation.ExactResult, error) {
	observation := runrecord.ServingObservation{
		Version: artifact.InitialDocumentVersion, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: time.Now().UnixNano(),
	}
	var walls processmeasure.Walls
	clock := processmeasure.NewStopwatch()
	staged, stages := processmeasure.WithStages(ctx)
	model, err := open(staged)
	load := walls.Elapsed(clock)
	fmt.Fprintf(said, "recipe: the load took %.3f s\n", time.Duration(load).Seconds())
	if _, reportErr := stages.Report(said, time.Duration(load)); reportErr != nil {
		fmt.Fprintf(said, "recipe: its parts are no division of it: %v\n", reportErr)
	}
	generation := &observedGeneration{servedModel: model, clock: processmeasure.NewStopwatch()}
	var result evaluation.ExactResult
	var total uint64
	failure, released := loadFailed, error(nil)
	if err == nil {
		generation.sample(ctx, runrecord.ServingHardwareStart)
		result, err = evaluation.Record(ctx, generation, request.Name, request.Prompt, request.MaxTokens)
		generation.sample(ctx, runrecord.ServingHardwareFinish)
		total, failure, released = generation.walls.Elapsed(generation.clock), generationFailed, model.Close()
		fmt.Fprintf(said, "recipe: from the request's start, the prompt was evaluated at %s, device memory was sampled at %s and the first token came at %s\n",
			time.Duration(generation.evaluated), time.Duration(generation.sampled), time.Duration(generation.first))
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
// device memory sampled at the stages of the request. It also keeps three
// moments of the request, each from its start: when the prompt was evaluated,
// when device memory had been sampled for it, and when the first token was
// emitted. A zero is a moment the request did not reach.
type observedGeneration struct {
	servedModel
	clock                     processmeasure.Stopwatch
	walls                     processmeasure.Walls
	prefill                   uint64
	evaluated, sampled, first uint64
	samples                   []runrecord.ServingHardwareSample
}

// Generate runs the model, timing its prefill and sampling device memory when
// the prompt is evaluated, and keeps the moment its first token is emitted.
// The token callback is wrapped only where the caller gave one: a generation
// with none takes another path through the model.
func (generation *observedGeneration) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	evaluated := options.OnPromptEvaluated
	options.OnPromptEvaluated = func(prompted inference.PromptEvaluation) {
		generation.prefill += uint64(max(prompted.Duration, 0))
		if generation.evaluated == 0 {
			generation.evaluated = generation.walls.Elapsed(generation.clock)
		}
		generation.sample(ctx, runrecord.ServingHardwarePrefill)
		if generation.sampled == 0 {
			generation.sampled = generation.walls.Elapsed(generation.clock)
		}
		if evaluated != nil {
			evaluated(prompted)
		}
	}
	if emitted := options.OnToken; emitted != nil {
		options.OnToken = func(event inference.TokenEvent) error {
			if generation.first == 0 {
				generation.first = generation.walls.Elapsed(generation.clock)
			}
			return emitted(event)
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
