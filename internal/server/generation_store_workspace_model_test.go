//go:build modeltest

package server

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestGenerationWorkspaceServesStoreMedia lists the lane store's media
// activations through the real catalog and generates one image with the
// cheapest of them, the host oscillator, publishing it as a PNG artifact.
// The lane store is only read: the generation runs in a scratch store over
// the oscillator's own weights, so a run adds no records to the store every
// other suite reads.
func TestGenerationWorkspaceServesStoreMedia(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "generating from store models is integration")
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatalf("store media unavailable: %v", err)
	}
	live, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatalf("store media unavailable: %v", err)
	}
	defer live.Close()
	ctx := t.Context()
	catalog := BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent)
	capabilities, err := NewStoreGenerationWorkspace(live, catalog, 256).WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil {
		t.Fatal(err)
	}
	var oscillator *WorkflowCapability
	tasks := map[recipe.Task]int{}
	for index, capability := range capabilities {
		tasks[capability.Task]++
		t.Logf("capability %s %s controls=%d refusal=%q", capability.Task, capability.Name, len(capability.Controls), capability.Refusal)
		if capability.Task == recipe.TaskImageGen && len(capability.Stages) > 0 && capability.Stages[0].Module.ID == modelrecipe.ModuleOscillatorImagePrepare {
			oscillator = &capabilities[index]
		}
	}
	if oscillator == nil {
		t.Fatal("store media unavailable: the store activates no oscillator image model")
	}
	if tasks[recipe.TaskSpeech] == 0 || tasks[recipe.TaskVideoGen] == 0 {
		t.Errorf("the store's speech and video activations are not listed: %v", tasks)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	activateModelAt(t, store, oscillator.Location, oscillatorImageDefinition)
	workspace := NewStoreGenerationWorkspace(store, catalog, 256)
	scratch, err := workspace.WorkflowCapabilities(ctx, workflowGeneration)
	if err != nil || len(scratch) != 1 || scratch[0].Refusal != "" {
		t.Fatalf("scratch capabilities = %+v, %v", scratch, err)
	}
	reporter := &recordingReporter{}
	completion, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, scratch[0], json.RawMessage(`{"class":1,"seed":424242}`), reporter)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, found, err := store.Artifact(ctx, completion.Outputs[0])
	if err != nil || !found || descriptor.MediaType != media.PNGMediaType {
		t.Fatalf("output artifact = %+v found=%v err=%v", descriptor, found, err)
	}
	// The runtime's own input document (the decoded request under the
	// capability's input schema) is the run's input, never a second record.
	request, err := requireRecordedRequest(ctx, store, completion.Run)
	if err != nil || !strings.HasSuffix(request.Schema, "-input.v1") {
		t.Fatalf("request record = %+v, %v", request, err)
	}
	t.Logf("generated %s from %s: %d bytes; request record %s", descriptor.ID, oscillator.Name, descriptor.Size, request.Schema)
	// Replay: the recorded request resubmitted unchanged answers with the
	// same output behind the same request document (a page's regenerate),
	// and the request with another seed answers with a different output.
	replayed, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, scratch[0], json.RawMessage(`{"class":1,"seed":424242}`), reporter)
	if err != nil {
		t.Fatal(err)
	}
	again, err := requireRecordedRequest(ctx, store, replayed.Run)
	if err != nil || again.ID != request.ID || replayed.Outputs[0] != completion.Outputs[0] {
		t.Fatalf("replay = %+v request %s, %v; want output %s behind %s", replayed, again.ID, err, completion.Outputs[0], request.ID)
	}
	varied, err := workspace.ExecuteWorkflow(ctx, workflowGeneration, scratch[0], json.RawMessage(`{"class":1,"seed":424243}`), reporter)
	if err != nil || varied.Outputs[0] == completion.Outputs[0] {
		t.Fatalf("varied = %+v, %v", varied, err)
	}
}
