package server

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/finding"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// TestServingFailureBecomesAFinding fails one attempt on each serving path,
// a generation the model refuses and an operation the server observes, and
// holds each failure to a failed serving observation under the activation's
// recipe, the record the scorecard's failure-rate cell counts, and an open
// finding that cites it and the recipe.
func TestServingFailureBecomesAFinding(t *testing.T) {
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
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
			return nil, errors.New("the model refused the class")
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	if _, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, capabilities[0], json.RawMessage(`{"class":1}`), &recordingReporter{}); err == nil {
		t.Fatal("the refused generation succeeded")
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	operationID, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskImageGen, Recipe: recipeID},
		func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
			return handler.executeObservedOperation(ctx, reporter, recipe.TaskImageGen, recipeID, modelID,
				func(context.Context, operation.Reporter) (operation.Completion, error) {
					return operation.Completion{}, errors.New("device lost")
				})
		})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := handler.operations.Wait(ctx, operationID); err != nil {
		t.Fatal(err)
	}
	var failed []artifact.ID
	for _, observation := range storedServingObservations(t, store) {
		if observation.Outcome == runrecord.OutcomeFailed && observation.Recipe == recipeID {
			failed = append(failed, observation.ID)
		}
	}
	var cited []artifact.ID
	if _, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: finding.MediaType, Schema: finding.Schema}},
		Order:     overgodb.DocumentOldestFirst,
	}, finding.Parse, func(_ overgodb.DocumentView, parked finding.Document) error {
		if parked.Status != finding.StatusOpen || !slices.Equal(parked.OwnerSurfaces, []artifact.ID{recipeID}) || len(parked.Evidence) != 1 {
			t.Errorf("finding %+v does not cite one failed attempt under recipe %s", parked, recipeID)
		}
		cited = append(cited, parked.Evidence...)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	sortIDs := func(ids []artifact.ID) { slices.SortFunc(ids, artifact.CompareID) }
	sortIDs(failed)
	sortIDs(cited)
	if len(failed) != 2 || !slices.Equal(failed, cited) {
		t.Fatalf("failed attempts %v under recipe %s, findings cite %v; want each of the two failures parked", failed, recipeID, cited)
	}
}
