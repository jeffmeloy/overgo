package server

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestGenerationRunPublishesServingObservation runs one image generation
// that succeeds and one that fails through the store generation workspace,
// and holds the serving observations the scorecard counts to what a user
// met: two attempts under the activation's recipe, one failed, and a
// success that carries its time to the whole output and the device memory
// it peaked at. The scorecard's own tests turn such observations into
// cells.
func TestGenerationRunPublishesServingObservation(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := activatedImageModel(t, store)
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskImageGen)
	if err != nil || !found {
		t.Fatalf("activation = (%v, %v)", found, err)
	}
	const (
		peak        = 3 << 20
		load        = 2 * time.Second
		decode      = 5 * time.Second
		outputToken = 7
	)
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			if raw == `{"class":2}` {
				return nil, errors.New("the model refused the class")
			}
			return capabilityruntime.Measured{Output: tinyPNG(t), PeakDeviceBytes: peak, Load: load, Decode: decode, OutputTokens: outputToken}, nil
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	reporter := &recordingReporter{}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(`{"class":1}`), reporter); err != nil {
		t.Fatal(err)
	}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(`{"class":2}`), reporter); err == nil {
		t.Fatal("the refused generation succeeded")
	}
	outcomes := map[runrecord.Outcome]int{}
	for _, observation := range storedServingObservations(t, store) {
		if observation.Model != modelID || observation.Recipe != activation.Definition.ID || observation.Task != recipe.TaskImageGen {
			t.Fatalf("observation %+v is not under the activation's model, recipe and task", observation)
		}
		outcomes[observation.Outcome]++
		if observation.Outcome == runrecord.OutcomeSucceeded && (observation.MeasuredNS == 0 || observation.Resources.PeakDeviceBytes != peak) {
			t.Fatalf("the success records %d ns and %d peak bytes, want the whole output's time and %d bytes",
				observation.MeasuredNS, observation.Resources.PeakDeviceBytes, peak)
		}
		// What the command line records of an attempt, a person at the page's attempt records too: the load, the
		// decode and the tokens the executor measured, which the load-time and throughput cells read.
		if observation.Outcome == runrecord.OutcomeSucceeded {
			phases := map[runrecord.Phase]uint64{}
			for _, phase := range observation.Phases {
				phases[phase.Phase] = phase.DurationNS
			}
			if phases[runrecord.PhaseLoad] != uint64(load) || phases[runrecord.PhaseDecode] != uint64(decode) || observation.Usage.OutputTokens != outputToken {
				t.Fatalf("the success records load %d ns, decode %d ns and %d tokens, want %d, %d and %d",
					phases[runrecord.PhaseLoad], phases[runrecord.PhaseDecode], observation.Usage.OutputTokens, uint64(load), uint64(decode), outputToken)
			}
		}
	}
	if outcomes[runrecord.OutcomeSucceeded] != 1 || outcomes[runrecord.OutcomeFailed] != 1 || len(outcomes) != 2 {
		t.Fatalf("observed outcomes %v, want one success and one failure", outcomes)
	}
}

// TestGenerationAttemptCitesTheRunItsExecutorMade runs three image
// generations through the store generation workspace: one whose executor
// reports a run bound to a device environment, one whose executor reports no
// run, and one whose executor fails having recorded its own failed run. It
// holds each attempt to the run and the environment that made it: the
// executor's run under that run's environment where one was reported, the
// workspace's own run under this process's environment where none was, and
// the executor's failed run for the failure. The operation's receipt stays
// the workspace's run each time, since that is the run that cites what the
// page published.
func TestGenerationAttemptCitesTheRunItsExecutorMade(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := activatedImageModel(t, store)
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskImageGen)
	if err != nil || !found {
		t.Fatalf("activation = (%v, %v)", found, err)
	}
	recipeID := activation.Definition.ID
	// The executor's run is bound to an environment that is not this
	// process's, so an attempt recorded under the observer's own environment
	// cannot pass for one recorded under its run's.
	device := testutil.ArtifactID(t, artifact.KindEvidence, "executor-environment")
	made := testutil.ArtifactID(t, artifact.KindOutput, "executor-output")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/executor/facts", Artifacts: []artifact.Descriptor{{ID: device}, {ID: made}}}); err != nil {
		t.Fatal(err)
	}
	inner, err := runrecord.NewBoundRun(recipeID, runrecord.OutcomeSucceeded, nil, []artifact.ID{made}, "", strings.Repeat("ab", 20), device, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The executor's failure code is not the workspace's, so the two failed
	// runs are different records.
	refused, err := runrecord.NewRun(recipeID, runrecord.OutcomeFailed, nil, nil, "executor_refused")
	if err != nil {
		t.Fatal(err)
	}
	for key, run := range map[string]runrecord.Run{"fixture/executor/run": inner, "fixture/executor/refused": refused} {
		batch, err := run.Batch(key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	process, err := runrecord.CurrentEnvironment("process", "go")
	if err != nil {
		t.Fatal(err)
	}
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			switch raw {
			case `{"class":1}`:
				return capabilityruntime.Measured{Output: tinyPNG(t), Run: inner.ID}, nil
			case `{"class":2}`:
				return capabilityruntime.Measured{Output: tinyPNG(t)}, nil
			}
			return capabilityruntime.Measured{Run: refused.ID}, errors.New("the model refused the class")
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	// attempt runs one request and returns its receipt beside the one
	// attempt it added to the store.
	attempts := 0
	attempt := func(raw string, wantErr bool) (artifact.ID, runrecord.ServingObservation) {
		t.Helper()
		completion, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(raw), &recordingReporter{})
		if (err != nil) != wantErr || completion.Run.Kind() != artifact.KindRun {
			t.Fatalf("request %s completed as %+v with error %v", raw, completion, err)
		}
		attempts++
		observed := storedServingObservations(t, store)
		if len(observed) != attempts {
			t.Fatalf("request %s left %d attempts in the store, want %d", raw, len(observed), attempts)
		}
		return completion.Run, observed[attempts-1]
	}

	// Each check reports and the test goes on, so one run shows every part
	// of the attribution that is wrong.
	receipt, observed := attempt(`{"class":1}`, false)
	if observed.Run != inner.ID || receipt == inner.ID {
		t.Errorf("the attempt cites run %s beside the receipt %s, want the executor's run %s and the workspace's own receipt", observed.Run, receipt, inner.ID)
	}
	if observed.Environment != device {
		t.Errorf("the attempt is under environment %s, want %s, the one its run is bound to", observed.Environment, device)
	}

	receipt, observed = attempt(`{"class":2}`, false)
	if observed.Run != receipt || observed.Environment != process.ID {
		t.Errorf("an executor that reported no run left an attempt citing %s under %s, want the workspace's run %s under this process's environment %s",
			observed.Run, observed.Environment, receipt, process.ID)
	}

	receipt, observed = attempt(`{"class":3}`, true)
	if observed.Outcome != runrecord.OutcomeFailed || observed.Run != refused.ID || receipt == refused.ID {
		t.Errorf("the failed attempt is %s citing run %s beside the receipt %s, want a failure citing the executor's failed run %s and the workspace's own receipt",
			observed.Outcome, observed.Run, receipt, refused.ID)
	}
	if observed.Environment != process.ID {
		t.Errorf("the failed attempt is under environment %s, want this process's %s: its run names none", observed.Environment, process.ID)
	}
}

// storedServingObservations is every serving observation the store holds,
// oldest first.
func storedServingObservations(t *testing.T, store *overgodb.Store) []runrecord.ServingObservation {
	t.Helper()
	var observations []runrecord.ServingObservation
	if _, err := overgodb.VisitDecodedDocuments(t.Context(), store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
		observations = append(observations, observation)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return observations
}
