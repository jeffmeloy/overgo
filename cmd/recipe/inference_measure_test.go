package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// The served fixture's prompt, its output and the device memory it reports.
const (
	servedContext      = 32
	servedPromptTokens = 3
	servedOutput       = "ok"
	servedCurrentBytes = 1 << 20
	servedPeakBytes    = 3 << 20
	servedAllocations  = 7
)

// servedFixture is a loaded model off the device: it evaluates every prompt as
// servedPromptTokens tokens and answers servedOutput, taking time the clock can
// measure, and reports fixed device memory.
type servedFixture struct {
	t       *testing.T
	prompts []string
}

// tick waits on the clock's own progress until its counter passes since, so
// a span the fixture takes is one the stopwatch can time, and returns it.
func tick(t *testing.T, since time.Duration) time.Duration {
	for {
		now, err := processmeasure.Counter()
		if err != nil {
			t.Fatal(err)
		}
		if now > since {
			return now
		}
	}
}

// Generate answers servedOutput a piece per token after evaluating the prompt,
// each step taking a tick of the clock.
func (fixture *servedFixture) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	fixture.prompts = append(fixture.prompts, prompt)
	started := tick(fixture.t, 0)
	evaluated := tick(fixture.t, started)
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: servedPromptTokens, Duration: evaluated - started})
	tick(fixture.t, evaluated)
	ids := make([]tokenizer.TokenID, servedPromptTokens, servedPromptTokens+len(servedOutput))
	for index, piece := range strings.Split(servedOutput, "") {
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(index), Piece: piece}); err != nil {
			return nil, "", err
		}
		ids = append(ids, tokenizer.TokenID(index))
	}
	return ids, servedOutput, nil
}

// DeviceMemoryStats reports the fixture's fixed device memory.
func (*servedFixture) DeviceMemoryStats(context.Context) (driver.MemoryStats, error) {
	return driver.MemoryStats{CurrentBytes: servedCurrentBytes, PeakBytes: servedPeakBytes, Allocations: servedAllocations}, nil
}

// Close releases nothing.
func (*servedFixture) Close() error { return nil }

// servingAttempts reads every serving attempt the store at directory holds,
// newest first.
func servingAttempts(t *testing.T, directory string) []runrecord.ServingObservation {
	t.Helper()
	reader, err := overgodb.OpenReadOnly(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	var observed []runrecord.ServingObservation
	if _, err := overgodb.VisitDecodedDocuments(t.Context(), reader, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema}},
		Order:     overgodb.DocumentNewestFirst,
	}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
		observed = append(observed, observation)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return observed
}

// TestInferenceMeasurementFillsTheServingCells holds one measurement of an
// active inference recipe to the observation the scorecard's serving cells
// read. The recipe is activated the way the library activates one, by an exact
// suite recorded and replayed, whose first case speaks the model's own
// alphabet; the measurement runs that case on the loaded model, and the
// attempt it records gives the row its first output, throughput, failure rate,
// load and peak device memory.
func TestInferenceMeasurementFillsTheServingCells(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	path := testutil.HermeticLlamaGGUF(t, servedContext)
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	candidate, err := prepareExactCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	cases := []evaluation.ExactCase{
		{Name: "dna", Prompt: "<dna>ATGGCG", Text: servedOutput, MaxTokens: len(servedOutput), PromptTokens: servedPromptTokens, GeneratedTokens: len(servedOutput)},
		{Name: "text", Prompt: "The capital of France is", Text: servedOutput, MaxTokens: len(servedOutput), PromptTokens: servedPromptTokens, GeneratedTokens: len(servedOutput)},
	}
	exact, err := evaluation.CompileExact(evaluation.ExactSuite{Schema: "test/v1", Source: "measured request", Cases: cases})
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := environment.Batch("fixture/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	revision := strings.Repeat("a", 40)
	plan, err := evaluation.BindExact(exact, evaluation.ExactAuthorities{
		ModelDefinition: candidate.Resolved.Document.ID, RuntimeRecipe: candidate.Definition.ID, CodeCommit: revision,
		Environment: environment.ID, Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	model := &servedFixture{t: t}
	report, err := evaluation.EvaluateExactSharded(ctx, store, model, exact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := modelintake.PublishVerification(ctx, store, candidate.Definition, revision, time.Millisecond, "cuda:0", "cuda",
		fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", plan.Identity(), report, len(cases)))
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, candidate.Definition, verification, recipe.EvidenceVerified, "recorded and replayed"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	result, err := measureInference(ctx, directory, path, func(context.Context) (servedModel, error) {
		tick(t, tick(t, 0))
		return model, nil
	})
	if err != nil || result.Text != servedOutput || model.prompts[len(model.prompts)-1] != cases[0].Prompt {
		t.Fatalf("measurement = %+v, %v after prompts %q; want the first verified case answered", result, err, model.prompts)
	}
	observed := servingAttempts(t, directory)
	if len(observed) != 1 || observed[0].Recipe != candidate.Definition.ID || observed[0].Outcome != runrecord.OutcomeSucceeded || observed[0].Usage.OutputTokens == 0 {
		t.Fatalf("serving attempts = %+v; want one succeeded attempt of the recipe with output tokens", observed)
	}
	// The scorecard's serving cells read these facts: the load and decode
	// phases, the prefill sample's elapsed time, and the peak device memory.
	phases := map[runrecord.Phase]bool{}
	for _, phase := range observed[0].Phases {
		phases[phase.Phase] = phase.DurationNS != 0
	}
	prefilled, peak := false, uint64(0)
	for _, sample := range observed[0].Hardware {
		prefilled = prefilled || sample.Stage == runrecord.ServingHardwarePrefill && sample.ElapsedNS != 0
		peak = max(peak, sample.DevicePeakBytes)
	}
	if !phases[runrecord.PhaseLoad] || !phases[runrecord.PhaseDecode] || !prefilled || peak != servedPeakBytes {
		t.Fatalf("attempt phases %+v, prefill sampled %t, peak %d bytes; want timed load and decode, a prefill sample and %d bytes", observed[0].Phases, prefilled, peak, servedPeakBytes)
	}
}
