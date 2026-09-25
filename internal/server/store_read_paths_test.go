package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestStoreReadsScaleWithTheirAnswer holds the operator timeline to the
// operation it names: it walks that operation's receipts from each node's
// alias back through every receipt before it, so a crowd of other operations'
// receipts is never read, and it answers exactly the receipts a scan of every
// receipt would have found for that operation, superseded states included, in
// the order the store introduced them.
func TestStoreReadsScaleWithTheirAnswer(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "timeline recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "timeline/recipe", Artifacts: []artifact.Descriptor{{ID: recipeID}}}); err != nil {
		t.Fatal(err)
	}
	// Each operation runs two nodes through admitted, running and completed; the second node retries.
	publish := func(name string) artifact.ID {
		t.Helper()
		operation := testutil.ArtifactID(t, artifact.KindEvidence, "timeline operation "+name)
		for index, node := range []recipe.NodeID{"prepare", "tool"} {
			receipt := runrecord.StageReceipt{Recipe: recipeID, Node: node, Operation: operation, Attempt: 1, State: runrecord.StageAdmitted}
			var descriptors []artifact.Descriptor
			if index == 0 {
				descriptors = []artifact.Descriptor{{ID: operation}}
			}
			states := []runrecord.StageState{runrecord.StageAdmitted, runrecord.StageRunning, runrecord.StageCompleted}
			if node == "tool" {
				states = append(states, runrecord.StageAdmitted, runrecord.StageRunning, runrecord.StageCompleted)
			}
			for step, state := range states {
				if step == 3 {
					receipt.Attempt++
				}
				receipt.State = state
				if _, err := runrecord.PublishStageReceipt(ctx, store, receipt, nil, descriptors); err != nil {
					t.Fatal(err)
				}
				descriptors = nil
			}
		}
		return operation
	}
	var target artifact.ID
	for index := range 12 {
		operation := publish(string(rune('a' + index)))
		if index == 5 {
			target = operation
		}
	}
	// What a scan of every receipt finds for the operation, in introduction order.
	var scanned []artifact.ID
	if _, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.StageReceiptMediaType, Schema: runrecord.StageReceiptSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, runrecord.ParseStageReceipt, func(_ overgodb.DocumentView, receipt runrecord.StageReceipt) error {
		if receipt.Operation == target {
			scanned = append(scanned, receipt.ID)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	handler := newTestHandlerForRepository(t, store, &fakeGenerator{})
	defer handler.Close()
	answer := serveTestRequest(handler, http.MethodGet, "/operations/timeline?id="+target.String(), "")
	var timeline struct {
		Events []runrecord.OperatorTimelineEvent `json:"events"`
	}
	if answer.Code != http.StatusOK || json.Unmarshal(answer.Body.Bytes(), &timeline) != nil {
		t.Fatalf("timeline status=%d body=%s", answer.Code, answer.Body)
	}
	var walked []artifact.ID
	for _, event := range timeline.Events {
		walked = append(walked, event.Artifact)
	}
	if len(scanned) != 9 || !slices.Equal(walked, scanned) {
		t.Fatalf("the timeline walked %v, a full scan finds %v", walked, scanned)
	}
}
