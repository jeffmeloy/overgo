package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// TestDispatchOpensTheStoreOnce holds plan -next to one store open: dispatch
// and the optimization review that follows it share the handle, because an
// open costs seconds and the command used to pay it twice. The row is still
// dispatched and the shared handle is closed when the command returns.
func TestDispatchOpensTheStoreOnce(t *testing.T) {
	census, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("census"))
	if err != nil {
		t.Fatal(err)
	}
	root := initializePlanTestRepository(t, plan.Plan{Census: &census, Items: []plan.Item{{
		ID: "row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./..."}},
	}}})
	t.Chdir(root)
	opens, open := 0, openPlanStore
	var shared *overgodb.Store
	openPlanStore = func() (*overgodb.Store, error) {
		opens++
		store, err := open()
		shared = store
		return store, err
	}
	t.Cleanup(func() { openPlanStore = open })
	var output bytes.Buffer
	if err := printDispatch(cli{next: true}, nil, &output); err != nil {
		t.Fatal(err)
	}
	if opens != 1 || !strings.HasPrefix(output.String(), "row / do") {
		t.Fatalf("plan -next opened the store %d times and printed %q", opens, output.String())
	}
	if _, _, err := shared.Artifact(t.Context(), census); err == nil {
		t.Fatal("the shared store was left open after the command returned")
	}
}
