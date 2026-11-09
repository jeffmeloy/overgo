package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func videoWorkspaceFixture(t *testing.T) (*Handler, *nativeMediaWorkspace, []byte) {
	t.Helper()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	video := []byte("mp4-output-bytes")
	videoID := testutil.ArtifactBytesID(t, artifact.KindOutput, video)
	generateRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "video-generate-recipe")
	generateRun := testutil.ArtifactID(t, artifact.KindRun, "video-generate-run")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "native/video/outputs",
		Artifacts: []artifact.Descriptor{{ID: generateRecipe}, {ID: generateRun}},
		Contents: []artifact.Content{
			{Descriptor: artifact.Descriptor{ID: videoID, Size: uint64(len(video)), MediaType: "video/mp4"}, Data: video},
		},
	}); err != nil {
		t.Fatal(err)
	}
	workspace := &nativeMediaWorkspace{
		fakeGenerator: &fakeGenerator{},
		capabilities: []WorkflowCapability{
			{
				Task: recipe.TaskVideoGen, Recipe: generateRecipe,
				Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.video"}}},
				Controls: []workflowControl{
					{Name: "prompt", Type: workflowControlText, Required: true},
					{Name: "format", Type: workflowControlText, Choices: []string{"gif", "mp4"}, Default: "mp4"},
				},
			},
		},
		completion: map[recipe.Task]operation.Completion{
			recipe.TaskVideoGen: {Run: generateRun, Outputs: []artifact.ID{videoID}},
		},
	}
	handler := testcheck.Must(New(Config{Repository: store, Workflows: workspace}, workspace))(t)
	t.Cleanup(func() {
		_ = handler.Close()
		_ = store.Close()
	})
	return handler, workspace, video
}

// TestVideoGenerationWorkspace pins the text-to-video protocol: a
// prompt runs the registered video-gen capability as one workflow
// operation, the output must be a committed video artifact, and its
// content URL serves the exact bytes as video/mp4.
func TestVideoGenerationWorkspace(t *testing.T) {
	t.Parallel()
	handler, workspace, video := videoWorkspaceFixture(t)
	recipeID := workspace.capabilities[0].Recipe
	response := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations",
		`{"model":"`+recipeID.String()+`","prompt":"a fox on a trail"}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result nativeImageResponse
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	if len(result.Data) != 1 {
		t.Fatalf("videos=%+v", result.Data)
	}
	content := serveTestRequest(handler, http.MethodGet, result.Data[0].URL, "")
	if content.Code != http.StatusOK || content.Header().Get("Content-Type") != "video/mp4" ||
		!bytes.Equal(content.Body.Bytes(), video) {
		t.Fatalf("content status=%d type=%q", content.Code, content.Header().Get("Content-Type"))
	}
	if len(workspace.executed) != 1 || workspace.executed[0] != recipe.TaskVideoGen {
		t.Fatalf("executed=%v", workspace.executed)
	}
}

// TestAVideoRequestIsGivenItsFormatsDefault holds the intake to the default a
// control declares: a clip asked for in no format, or in a null one, runs with
// the control's "mp4" in its request, so the request recorded names what ran,
// and one that names "gif" keeps it. A default outside its control's choices,
// or on a control that is no text, is refused with the capability.
func TestAVideoRequestIsGivenItsFormatsDefault(t *testing.T) {
	t.Parallel()
	handler, workspace, _ := videoWorkspaceFixture(t)
	recipeID := workspace.capabilities[0].Recipe.String()
	for body, want := range map[string]string{
		`{"model":"` + recipeID + `","prompt":"a fox"}`:                 "mp4",
		`{"model":"` + recipeID + `","prompt":"a fox","format":null}`:   "mp4",
		`{"model":"` + recipeID + `","prompt":"a fox","format":"gif"}`:  "gif",
		`{"model":"` + recipeID + `","prompt":"an owl","format":"mp4"}`: "mp4",
	} {
		response := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations", body)
		if response.Code != http.StatusOK {
			t.Fatalf("%s: status=%d body=%s", body, response.Code, response.Body.String())
		}
		workspace.mu.Lock()
		input := workspace.inputs[len(workspace.inputs)-1]
		workspace.mu.Unlock()
		var ran struct {
			Format string `json:"format"`
		}
		testcheck.NoError(t, json.Unmarshal(input, &ran))
		if ran.Format != want {
			t.Fatalf("%s ran with %s, want the format %q", body, input, want)
		}
	}
	for _, control := range []workflowControl{
		{Name: "format", Type: workflowControlText, Choices: []string{"gif", "mp4"}, Default: "webm"},
		{Name: "steps", Type: workflowControlInteger, Default: "4"},
	} {
		capability := workspace.capabilities[0]
		capability.Controls = []workflowControl{control}
		if err := validateWorkflowCapabilities([]WorkflowCapability{capability}); err == nil {
			t.Fatalf("a control %+v was admitted", control)
		}
	}
}
