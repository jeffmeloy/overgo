package controlleraction

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestEvidenceDerivedSelection pins controlleraction's half of selection: only
// the typed evidence request crosses the action boundary, and the one derived
// decision must publish with its causal context against that evidence head.
func TestEvidenceDerivedSelection(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close driver decision store: %v", err)
		}
	})
	root := testutil.ArtifactID(t, artifact.KindEvidence, "driver publication root")
	testutil.PublishArtifact(t, store, root)
	causal, err := runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, root)
	if err != nil {
		t.Fatal(err)
	}
	head, _ := store.Head()
	if !head.Valid() {
		t.Fatalf("driver publication head = %s", head)
	}
	request := evaluation.EvidenceDriverRequest{
		Goal:              testutil.ArtifactID(t, artifact.KindRecipe, "driver publication goal"),
		Causal:            causal,
		Head:              head,
		Incumbent:         testutil.ArtifactID(t, artifact.KindEvidence, "driver publication incumbent probe"),
		InteractionBudget: testutil.ArtifactID(t, artifact.KindEvidence, "driver publication interaction budget"),
		ResourceBudget:    testutil.ArtifactID(t, artifact.KindEvidence, "driver publication resource budget"),
	}
	action := Action{
		Version: ActionVersion, Kind: KindDriverDecision,
		Driver: &DriverDecisionAction{Request: request},
	}
	encoded, err := json.Marshal(action)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseAction(encoded)
	if err != nil || parsed.Driver.Request.Goal != request.Goal || parsed.Driver.Request.Head != head {
		t.Fatalf("typed driver action round trip = (%+v, %v)", parsed, err)
	}
	request = parsed.Driver.Request
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	driver := document["driver"].(map[string]any)
	driver["facts"] = driver["request"]
	delete(driver, "request")
	legacy, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseAction(legacy); err == nil {
		t.Fatal("raw driver-decision facts bypass remained representable")
	}

	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.controller-driver-test+json",
		Schema: "overgo/controller-driver-test/v1",
	}
	content, err := contract.ContentBytes([]byte(`{"version":1}`))
	if err != nil {
		t.Fatal(err)
	}
	newBatch := func() artifact.Batch {
		t.Helper()
		batch, batchErr := artifact.NewDocumentBatch("controller-action/driver-test", []artifact.Content{content}, nil, nil)
		if batchErr != nil {
			t.Fatal(batchErr)
		}
		if batchErr = bindDriverDecisionAuthority(&batch, batch.Contents, request); batchErr != nil {
			t.Fatal(batchErr)
		}
		return batch
	}

	batch := newBatch()
	if len(batch.Contents) != 1 || len(batch.Causality) != 1 || batch.Causality[0].Root != root ||
		batch.Causality[0].Execution != content.Descriptor.ID ||
		batch.ExpectedHead == nil || *batch.ExpectedHead != head {
		t.Fatalf("driver publication authority = %+v", batch)
	}
	if _, found, err := store.Artifact(t.Context(), content.Descriptor.ID); err != nil || found {
		t.Fatalf("driver decision visible before commit = (%t, %v)", found, err)
	}
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindEvidence, "driver concurrent write"))
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err == nil {
		t.Fatal("stale evidence-derived decision committed after the repository head advanced")
	}
	request.Head, _ = store.Head()
	if !request.Head.Valid() {
		t.Fatalf("fresh driver publication head = %s", request.Head)
	}
	batch = newBatch()
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatalf("fresh evidence-derived decision did not publish: %v", err)
	}
	if _, found, err := store.Artifact(t.Context(), content.Descriptor.ID); err != nil || !found {
		t.Fatalf("committed driver decision is absent = (%t, %v)", found, err)
	}
}
