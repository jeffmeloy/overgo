package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingattempt"
	"overgo/internal/testutil"
	"overgo/internal/textgeneration"
	"overgo/internal/workflowruntime"
)

func TestExactVerificationCandidateBinding(t *testing.T) {
	path := testutil.HermeticLlamaGGUF(t, 32)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := prepareExactCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative); err == nil {
		t.Fatal("missing registered profile accepted")
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	want, err := modelintake.PrepareInferenceCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	check := func() {
		t.Helper()
		candidate, err := prepareExactCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
		if err != nil {
			t.Fatal(err)
		}
		loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved)
		if err != nil {
			t.Fatal(err)
		}
		defer loaded.Close()
		identity, err := loaded.Identity()
		if err != nil || candidate.Definition.ID != want.Definition.ID || identity.Recipe != want.Definition.ID ||
			identity.Profile != want.Resolved.Profile.ID || identity.Definition != want.Resolved.Document.ID {
			t.Fatalf("candidate binding differs: %+v, %v", identity, err)
		}
	}
	check()
	if _, active, err := modelrecipe.ActiveRecord(ctx, store, want.Inventory.Manifest.ID, recipe.TaskInference); err != nil || active {
		t.Fatalf("verification activated a candidate: %v", err)
	}
	// A prior active recipe deliberately uses a different decode session.
	prior, err := modelintake.PrepareInferenceCandidate(ctx, store, path,
		modelintake.SessionOverride{Set: true, Value: modelrecipe.DecodeSessionRequest}, recipe.ResidencyHybridNative)
	if err != nil || prior.Definition.ID == want.Definition.ID {
		t.Fatalf("distinct active fixture: %v", err)
	}
	if err := modelrecipetest.PublishActivation(ctx, store, "fixture/prior-active", prior.Definition); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	check()
	check()
	active, found, err := modelrecipe.ActiveRecord(ctx, store, prior.Inventory.Manifest.ID, recipe.TaskInference)
	if err != nil || !found || active.Definition.ID != prior.Definition.ID {
		t.Fatalf("verification replaced the active recipe: %v", err)
	}
	canceled, cancel := context.WithCancelCause(ctx)
	cancel(context.Canceled)
	if _, err := prepareExactCandidate(canceled, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
	if got, ordinal := store.Head(); got != head || ordinal != sequence {
		t.Fatal("repeat or canceled preparation mutated the store")
	}
}

func TestPrepareCapabilityReusesPublishedFacts(t *testing.T) {
	path := testutil.TempGGUF(t, "facts.gguf", nil, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, binary.Size(float32(0)))),
	}})
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	capability := mediacapability.Capability{Resolve: func(string) (mediacapability.Source, error) {
		return mediacapability.Source{Inventory: inventory}, nil
	}}
	want, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	batch, err := inventory.Batch("fixture/another-publisher")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	unchanged := func() {
		t.Helper()
		if gotHead, gotSequence := store.Head(); gotHead != head || gotSequence != sequence {
			t.Fatalf("publication changed head/sequence: %s/%d, want %s/%d", gotHead, gotSequence, head, sequence)
		}
	}
	for _, phase := range []string{"foreign publisher", "repeat", "reopen"} {
		t.Run(phase, func(t *testing.T) {
			if phase == "reopen" {
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				store, err = overgodb.Open(directory)
				if err != nil {
					t.Fatal(err)
				}
			}
			model, definition, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability)
			if err != nil || model != inventory.Manifest.ID || definition.ID != want.ID {
				t.Fatalf("prepare = %s/%s, %v; want %s/%s", model, definition.ID, err, inventory.Manifest.ID, want.ID)
			}
			unchanged()
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := prepareCapability(ctx, store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled preparation = %v", err)
	}
	unchanged()
	inventory.Components[0].MediaType = "application/octet-stream"
	if _, _, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, overgodb.ErrArtifactConflict) {
		t.Fatalf("conflicting facts = %v", err)
	}
	unchanged()
}

// TestModelFactsFollowTheModel: a model's facts change while its identity
// stands. Its bytes move to another directory, or the code derives another
// fact from them. Both facts a projection verify publishes, the capability's
// and the inference candidate's, publish each change, where a key naming only
// the model or its definition refused the moved model as a conflict, and the
// store then answers for the new location, the new fact and the component
// group by identity.
func TestModelFactsFollowTheModel(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	original := testutil.HermeticLlamaGGUF(t, 32)
	data, err := os.ReadFile(original)
	if err != nil {
		t.Fatal(err)
	}
	moved := filepath.Join(t.TempDir(), filepath.Base(original))
	if err := os.WriteFile(moved, data, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, step := range []struct{ path, derived string }{
		{original, "first derivation"}, {moved, "first derivation"}, {moved, "second derivation"},
	} {
		file, err := gguf.Open(step.path)
		if err != nil {
			t.Fatal(err)
		}
		inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
		if err := errors.Join(err, file.Close()); err != nil {
			t.Fatal(err)
		}
		fact, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/fixture-derived-fact/v1"), struct{ Derived string }{step.derived})
		if err != nil {
			t.Fatal(err)
		}
		group, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
			Role: artifact.ComponentWeights, Name: "group/weights", Artifact: inventory.Components[0].ID,
		}})
		if err != nil {
			t.Fatal(err)
		}
		capability := mediacapability.Capability{Resolve: func(string) (mediacapability.Source, error) {
			return mediacapability.Source{Inventory: inventory, Manifests: []artifact.Manifest{group},
				Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
					definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, modelID)
					return definition, []artifact.Content{fact}, err
				}}, nil
		}}
		if _, _, err := prepareCapability(ctx, store, step.path, recipe.TaskSeq2Seq, capability); err != nil {
			t.Fatalf("%s at %s: capability facts: %v", step.derived, step.path, err)
		}
		if _, err := prepareExactCandidate(ctx, store, step.path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative); err != nil {
			t.Fatalf("%s at %s: candidate facts: %v", step.derived, step.path, err)
		}
		for _, location := range inventory.Locations {
			held, err := store.Locations(ctx, location.Artifact)
			if err != nil || !slices.Contains(held, location) {
				t.Fatalf("%s at %s: store holds %v for %s, want %v: %v", step.derived, step.path, held, location.Artifact, location, err)
			}
		}
		if held, err := store.HasContent(ctx, fact.Descriptor.ID); err != nil || !held {
			t.Fatalf("%s: derived fact %s absent: %v", step.derived, fact.Descriptor.ID, err)
		}
		if _, held, err := store.Manifest(ctx, group.ID); err != nil || !held {
			t.Fatalf("component group %s absent: %v", group.ID, err)
		}
	}
}

func TestCommandsRegisterExecutableCapabilities(t *testing.T) {
	for _, task := range []recipe.Task{
		recipe.TaskGeneration, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen, recipe.TaskVideoGen,
	} {
		t.Run(string(task), func(t *testing.T) {
			capability, ok := mediacapability.Catalog[task]
			if !ok || capability.Resolve == nil || capability.Execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
		})
	}
	projection := mediacapability.Projection(t.Context(), nil, "")
	if projection.Resolve == nil || projection.Execute != nil {
		t.Fatalf("projection capability = %+v", projection)
	}
}

func TestCapabilityVerificationActivatesCandidate(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "recipe-verifier-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, modelID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/recipe-verifier/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, "0123456789abcdef0123456789abcdef01234567", time.Millisecond,
		"host", "go", "fixture output validated",
	)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Gate.Recipe != definition.ID || verified.Run.Recipe != definition.ID {
		t.Fatalf("verification subject = %s/%s, want %s", verified.Gate.Recipe, verified.Run.Recipe, definition.ID)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceVerified, "fixture candidate execution",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskSeq2Seq)
	if err != nil {
		t.Fatal(err)
	}
	if !active || activation.Definition.ID != definition.ID {
		t.Fatalf("active = %t/%s, want true/%s", active, activation.Definition.ID, definition.ID)
	}
}

// TestMeasurementNeedsExecutableInput: measurement runs the request its
// activation's accepted verification retained whole, once. A verification
// that kept only the request's digest, or content that is no request, is a
// gap reported before the executor runs or any serving attempt is recorded,
// and a request another run of the recipe retained never stands in for it.
func TestMeasurementNeedsExecutableInput(t *testing.T) {
	whole := measuredContent(t, "overgo.fixture-input.v1", `{"prompt":"verified"}`)
	unrelated := measuredContent(t, "overgo.fixture-input.v1", `{"prompt":"another run's"}`)
	inventory := measuredModel(t)
	for name, test := range map[string]struct {
		accepted artifact.Content
		want     string
		cause    error
	}{
		"failed execution": {accepted: whole, want: string(whole.Data), cause: errors.New("executor failed")},
		"whole request":    {accepted: whole, want: string(whole.Data)},
		"digest only":      {accepted: measuredContent(t, "overgo.fixture-input-digest.v1", `{"request_sha256":"x","request_bytes":2000000}`)},
		"not a request":    {accepted: measuredContent(t, "overgo.fixture-output.v1", `{"prompt":"output"}`)},
		"nothing at all":   {},
	} {
		t.Run(name, func(t *testing.T) {
			directory := t.TempDir()
			handoff := filepath.Join(directory, "attempt.json")
			t.Setenv(servingattempt.OutputEnvironment, handoff)
			// Another run of the recipe retained a whole request; it is not
			// the accepted verification's.
			head, sequence := activateMeasured(t, directory, recipe.TaskSeq2Seq, inventory, unrelated, test.accepted)
			var ran []string
			capability := mediacapability.Capability{
				Resolve: func(string) (mediacapability.Source, error) {
					return mediacapability.Source{Inventory: inventory}, nil
				},
				Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
					ran = append(ran, input)
					return "measured", test.cause
				},
			}
			err := measureCapability(directory, "fixture-path", recipe.TaskSeq2Seq, capability)
			reopened, openErr := overgodb.Open(directory)
			if openErr != nil {
				t.Fatal(openErr)
			}
			defer reopened.Close()
			after, afterSequence := reopened.Head()
			if test.want != "" {
				if !errors.Is(err, test.cause) || !slices.Equal(ran, []string{test.want}) || afterSequence == sequence {
					t.Fatalf("measurement ran %q (%v), recorded attempt %v; want the verified request once, recorded", ran, err, afterSequence != sequence)
				}
				var id artifact.ID
				if err := jsonfile.DecodeStrict(handoff, &id); err != nil {
					t.Fatal(err)
				}
				observation, err := runrecord.RequireServingObservation(t.Context(), reopened, id)
				wantOutcome := runrecord.OutcomeSucceeded
				if test.cause != nil {
					wantOutcome = runrecord.OutcomeFailed
				}
				if err != nil || observation.Model != inventory.Manifest.ID || observation.Outcome != wantOutcome {
					t.Fatalf("handoff %s: %+v (%v)", id, observation, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "retained no runnable request") || len(ran) != 0 || after != head || afterSequence != sequence {
				t.Fatalf("measurement ran %q (%v), store moved %v; want a gap before execution", ran, err, after != head)
			}
		})
	}
}

// TestCapabilityMeasurementRecordsLoadAndDevicePeak holds the attempt a
// measured capability records to what its scorecard row's load and peak
// memory cells read: the load its executor's measurement carries, as the
// attempt's load phase within its whole wall, and the device memory the
// attempt peaked at. An execution that loaded no model, as a resident
// session's, records no load phase. What the process peaked at is the
// attempt's peak, over the figure its executor gives of its own model, which
// stands only where the process allocated nothing. Where neither holds any,
// an attempt that succeeded having loaded its model measured zero, and
// records it as a sample of the device at its finish; one that failed, or
// served a recorded answer and loaded nothing, records no peak at all.
func TestCapabilityMeasurementRecordsLoadAndDevicePeak(t *testing.T) {
	const executorPeak, processPeak = 3 << 30, 5 << 30
	failed := errors.New("fixture execution failed")
	inventory := measuredModel(t)
	verified := measuredContent(t, "overgo.fixture-input.v1", `{"prompt":"verified"}`)
	for name, test := range map[string]struct {
		loads             bool
		process, executor uint64
		cause             error
		peak              uint64
		zero              bool
	}{
		"a model loaded":                    {loads: true, executor: executorPeak, peak: executorPeak},
		"a resident session":                {executor: executorPeak, peak: executorPeak},
		"sessions resident together":        {loads: true, process: processPeak, executor: executorPeak, peak: processPeak},
		"a failure that allocated":          {loads: true, process: processPeak, cause: failed, peak: processPeak},
		"a model on the host":               {loads: true, zero: true},
		"a recorded answer, nothing loaded": {},
		"a failure on the host":             {loads: true, cause: failed},
	} {
		t.Run(name, func(t *testing.T) {
			// Tests share one process, whose own ledger a device test
			// before this one leaves raised.
			previous := processDevicePeak
			processDevicePeak = func() uint64 { return test.process }
			t.Cleanup(func() { processDevicePeak = previous })
			outcome := runrecord.OutcomeSucceeded
			if test.cause != nil {
				outcome = runrecord.OutcomeFailed
			}
			directory := t.TempDir()
			activateMeasured(t, directory, recipe.TaskSeq2Seq, inventory, verified)
			var load time.Duration
			capability := mediacapability.Capability{
				Resolve: func(string) (mediacapability.Source, error) {
					return mediacapability.Source{Inventory: inventory}, nil
				},
				Execute: func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
					// The load and the run each take a span the clock can time.
					started := tick(t, 0)
					ready := tick(t, started)
					if test.loads {
						load = ready - started
					}
					tick(t, ready)
					return capabilityruntime.Measured{Output: "measured", Load: load, PeakDeviceBytes: test.executor}, test.cause
				},
			}
			if err := measureCapability(directory, "fixture-path", recipe.TaskSeq2Seq, capability); !errors.Is(err, test.cause) {
				t.Fatalf("measurement = %v, want %v", err, test.cause)
			}
			observed := servingAttempts(t, directory)
			if len(observed) != 1 || observed[0].Task != recipe.TaskSeq2Seq || observed[0].Outcome != outcome {
				t.Fatalf("serving attempts = %+v; want one %s attempt of the capability", observed, outcome)
			}
			attempt := observed[0]
			var phases []runrecord.PhaseMetric
			if test.loads {
				phases = []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: uint64(load)}}
			}
			var samples []runrecord.ServingHardwareSample
			if test.zero {
				samples = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwareFinish, ElapsedNS: attempt.MeasuredNS}}
			}
			if !slices.Equal(attempt.Phases, phases) || uint64(load) >= attempt.MeasuredNS {
				t.Fatalf("attempt phases %+v within %d ns; want phases %+v", attempt.Phases, attempt.MeasuredNS, phases)
			}
			if attempt.Resources.PeakDeviceBytes != test.peak || !slices.Equal(attempt.Hardware, samples) {
				t.Fatalf("attempt peak %d bytes, device samples %+v; want %d bytes and samples %+v", attempt.Resources.PeakDeviceBytes, attempt.Hardware, test.peak, samples)
			}
		})
	}
}

// TestCapabilityMeasurementRunsTheModel holds a measurement to running the
// model. The store keeps the stage a request ran, and serving the identical
// request again recovers it: two runs of the request run the model's one
// node once. A measurement of that request recovers nothing, so the attempt
// it records times the model and not the store's replay: each of two
// measurements runs the node. Serving the request afterwards still recovers,
// and serving it fresh runs the node, as its own operation that leaves the
// request's replay as it was. The forecast is an output returned whole: no
// attempt records a decode.
func TestCapabilityMeasurementRunsTheModel(t *testing.T) {
	const request = `[1,2,3]`
	inventory := measuredModel(t)
	directory := t.TempDir()
	activateMeasured(t, directory, recipe.TaskForecast, inventory, measuredContent(t, "overgo.forecast-input.v1", request))
	var runs atomic.Int32
	var terminal error
	capability := mediacapability.Capability{
		Resolve: func(string) (mediacapability.Source, error) {
			return mediacapability.Source{Inventory: inventory}, nil
		},
		Execute: capabilityruntime.JSONScalar[[]float32, int, []float32](
			"forecast", func([]float32) error { return nil },
			func(context.Context, artifact.Repository, string, recipe.Program, []float32) (int, error) {
				return 1, nil
			},
			func(runtime *workflowruntime.Runtime, modelID artifact.ID, _ int) error {
				return workflowruntime.RegisterJSONStage[[]float32, []float32](
					runtime, modelrecipe.ModuleForecastSeries, modelID, artifact.JSONContract(artifact.KindOutput, "overgo.fixture-forecast.v1"),
					func(series []float32) ([]float32, error) {
						runs.Add(1)
						return series, terminal
					},
				)
			},
		),
	}
	for index, step := range []struct {
		measure, fresh bool
		runs           int32
	}{{runs: 1}, {runs: 1}, {measure: true, runs: 2}, {measure: true, runs: 3}, {runs: 3}, {fresh: true, runs: 4}, {runs: 4}} {
		var err error
		if step.measure {
			err = measureCapability(directory, "fixture-path", recipe.TaskForecast, capability)
		} else {
			err = executeCapability(directory, "fixture-path", "", "", modelrecipe.SessionWarm, recipe.TaskForecast, capability, request, step.fresh)
		}
		if err != nil || runs.Load() != step.runs {
			t.Fatalf("execution %d (measurement: %t, fresh: %t) left the model's node run %d times (%v); want %d: a served request recovers its stored stage, a measurement or a fresh run runs the model",
				index, step.measure, step.fresh, runs.Load(), err, step.runs)
		}
	}
	attempts := servingAttempts(t, directory)
	if len(attempts) != 7 {
		t.Fatalf("serving attempts = %d; want each of the seven executions recorded", len(attempts))
	}
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	input := measuredContent(t, "overgo.forecast-input.v1", request)
	output, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo.fixture-forecast.v1"), []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	for _, attempt := range attempts {
		run, err := runrecord.RequireExactRun(t.Context(), store, attempt.Run)
		if err != nil || run.Recipe != attempt.Recipe || run.Outcome != runrecord.OutcomeSucceeded || !slices.Contains(run.Inputs, input.Descriptor.ID) || !slices.Equal(run.Outputs, []artifact.ID{output.Descriptor.ID}) {
			t.Fatalf("serving attempt %s cites run %s = %+v (%v); want its persisted recipe, request and output", attempt.ID, attempt.Run, run, err)
		}
	}
	// Each deliberate measurement has its own operation. Failed/cancelled
	// executions must record the actual failed run, with no successful output.
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, cause := range []error{errors.New("fixture forecast failed"), context.Canceled} {
		terminal = cause
		if err := measureCapability(directory, "fixture-path", recipe.TaskForecast, capability); !errors.Is(err, cause) {
			t.Fatalf("terminal measurement = %v; want %v", err, cause)
		}
	}
	attempts = servingAttempts(t, directory)
	if len(attempts) != 9 || runs.Load() != 6 {
		t.Fatalf("terminal attempts=%d, executed nodes=%d; want nine attempts and six executions", len(attempts), runs.Load())
	}
	store, err = overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, attempt := range attempts {
		if attempt.Outcome == runrecord.OutcomeSucceeded {
			continue
		}
		run, err := runrecord.RequireExactRun(t.Context(), store, attempt.Run)
		if err != nil || run.Recipe != attempt.Recipe || run.Outcome != attempt.Outcome || !slices.Contains(run.Inputs, input.Descriptor.ID) || len(run.Outputs) != 0 {
			t.Fatalf("terminal attempt %s cites %+v (%v); want its actual recipe/request/outcome without output", attempt.ID, run, err)
		}
	}
	// A forecast is returned whole by a model that decodes no tokens: no
	// attempt of it, replayed or run, has a decode phase or output tokens.
	for _, attempt := range attempts {
		if attempt.Usage.OutputTokens != 0 || decodePhase(attempt) != 0 {
			t.Fatalf("attempt of a whole output has a %d ns decode phase and %d output tokens; want neither", decodePhase(attempt), attempt.Usage.OutputTokens)
		}
	}
}

// decodePhase is the wall of an attempt's decode phase, zero without one.
func decodePhase(attempt runrecord.ServingObservation) uint64 {
	for _, phase := range attempt.Phases {
		if phase.Phase == runrecord.PhaseDecode {
			return phase.DurationNS
		}
	}
	return 0
}

// decodingModel is a fixture seq2seq model that reports the tokens its last
// selection decoded, as a token-decoding model runtime does.
type decodingModel struct{ selected uint64 }

func (model *decodingModel) OutputTokens() uint64 { return model.selected }

// TestCapabilityMeasurementRecordsDecodeAndOutputTokens holds the attempt a
// measured capability records to what its scorecard row's throughput cell
// reads: output tokens over a decode phase. A seq2seq model runs the three
// stages of its recipe, encode, prepare and select, each over a span the
// clock can time, and reports the tokens its selection decoded: the attempt
// carries that count, and as its decode phase the wall of the select stage
// alone, within the attempt's whole wall.
func TestCapabilityMeasurementRecordsDecodeAndOutputTokens(t *testing.T) {
	const tokens = 4
	inventory := measuredModel(t)
	directory := t.TempDir()
	activateMeasured(t, directory, recipe.TaskSeq2Seq, inventory, measuredContent(t, "overgo.seq2seq-input.v1", `{"text":"source","max_tokens":4}`))
	timed := func() { tick(t, tick(t, 0)) }
	execute := capabilityruntime.JSONScalar[textgeneration.Request, *decodingModel, string](
		"seq2seq", textgeneration.Validate,
		func(context.Context, artifact.Repository, string, recipe.Program, textgeneration.Request) (*decodingModel, error) {
			return &decodingModel{}, nil
		},
		func(runtime *workflowruntime.Runtime, modelID artifact.ID, model *decodingModel) error {
			return workflowruntime.RegisterJSONPipeline(
				runtime, modelID, artifact.JSONContract(artifact.KindFile, "overgo.fixture-seq2seq-text.v1"),
				modelrecipe.ModuleSeq2SeqEncode, func(request textgeneration.Request) (int, error) {
					timed()
					return request.MaxTokens, nil
				},
				modelrecipe.ModuleSeq2SeqPrepare, func(limit int) (int, error) {
					timed()
					return limit, nil
				},
				modelrecipe.ModuleSeq2SeqSelect, func(limit int) (string, error) {
					timed()
					model.selected = uint64(limit)
					return "selected", nil
				},
			)
		},
	)
	// The walls the executor measured, by the node that took them.
	walls := map[recipe.NodeID]uint64{}
	capability := mediacapability.Capability{
		Resolve: func(string) (mediacapability.Source, error) {
			return mediacapability.Source{Inventory: inventory}, nil
		},
		Execute: func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
			output, err := execute(ctx, store, path, selection, input)
			measured, _ := output.(capabilityruntime.Measured)
			for _, wall := range measured.Phases {
				walls[wall.Node] = wall.WallNS
			}
			return output, err
		},
	}
	if err := measureCapability(directory, "fixture-path", recipe.TaskSeq2Seq, capability); err != nil {
		t.Fatal(err)
	}
	observed := servingAttempts(t, directory)
	if len(observed) != 1 || observed[0].Task != recipe.TaskSeq2Seq || observed[0].Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("serving attempts = %+v; want one succeeded attempt of the capability", observed)
	}
	if len(walls) != 3 || walls["encode"] == 0 || walls["prepare"] == 0 || walls["select"] == 0 {
		t.Fatalf("executor node walls = %v; want the three stages timed", walls)
	}
	attempt, decode := observed[0], decodePhase(observed[0])
	if attempt.Usage.OutputTokens != tokens || decode != walls["select"] || decode >= attempt.MeasuredNS {
		t.Fatalf("attempt has %d output tokens over a %d ns decode phase within %d ns; want %d tokens over the select stage's %d ns alone (node walls %v)",
			attempt.Usage.OutputTokens, decode, attempt.MeasuredNS, tokens, walls["select"], walls)
	}
}

// measuredContent is a document of schema holding body: a run's retained
// request when the schema is a request's.
func measuredContent(t *testing.T, schema, body string) artifact.Content {
	t.Helper()
	content, err := artifact.JSONContract(artifact.KindFile, schema).ContentBytes([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	return content
}

// measuredModel inventories a model of one tensor.
func measuredModel(t *testing.T) modelartifact.Inventory {
	t.Helper()
	path := testutil.TempGGUF(t, "measured.gguf", nil, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, binary.Size(float32(0)))),
	}})
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	return inventory
}

// activateMeasured gives the model an active recipe of task in a new store
// at directory: each of runs is a verification of the recipe that recorded
// that request, in order, and its activation accepts the last. It returns
// the head of the store it closed.
func activateMeasured(t *testing.T, directory string, task recipe.Task, inventory modelartifact.Inventory, runs ...artifact.Content) (artifact.CommitID, uint64) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := inventory.Batch("fixture/measured/model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(task, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/measured/candidate", definition); err != nil {
		t.Fatal(err)
	}
	var accepted modelrecipe.Verification
	for _, input := range runs {
		accepted, err = modelintake.PublishMeasuredVerification(ctx, store, definition,
			"0123456789abcdef0123456789abcdef01234567", time.Millisecond, "host", "go", "fixture output validated",
			capabilityruntime.Measured{Input: input})
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, accepted, recipe.EvidenceVerified, "fixture candidate execution"); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return head, sequence
}
