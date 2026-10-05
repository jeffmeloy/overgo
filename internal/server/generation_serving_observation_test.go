package server

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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
