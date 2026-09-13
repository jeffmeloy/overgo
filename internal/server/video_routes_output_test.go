package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// videoOutputFixture serves one video-gen capability whose operation
// commits an output of the given media type, the way the executors do.
func videoOutputFixture(t *testing.T, mediaType string, output []byte) (*Handler, artifact.ID) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	outputID := testutil.ArtifactBytesID(t, artifact.KindOutput, output)
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "video-generate-"+mediaType)
	runID := testutil.ArtifactID(t, artifact.KindRun, "video-generate-run-"+mediaType)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "native/video/declared-output/" + mediaType,
		Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: runID}},
		Contents:  []artifact.Content{{Descriptor: artifact.Descriptor{ID: outputID, Size: uint64(len(output)), MediaType: mediaType}, Data: output}},
	}); err != nil {
		t.Fatal(err)
	}
	workspace := &nativeMediaWorkspace{
		fakeGenerator: &fakeGenerator{},
		capabilities: []WorkflowCapability{{
			Task: recipe.TaskVideoGen, Recipe: recipeID,
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.video"}}},
			Controls: []WorkflowControl{{Name: "prompt", Type: WorkflowControlText, Required: true}},
		}},
		completion: map[recipe.Task]operation.Completion{recipe.TaskVideoGen: {Run: runID, Outputs: []artifact.ID{outputID}}},
	}
	handler, err := New(Config{Repository: store}, workspace)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = handler.Close()
		_ = store.Close()
	})
	return handler, recipeID
}

// TestVideoRoutesDeliverDeclaredOutput: the native video route delivers
// the output the executors declare, GIF-encoded video, beside a container
// video type; an output that is neither is refused; and the retired
// video-edit route and mode are gone.
func TestVideoRoutesDeliverDeclaredOutput(t *testing.T) {
	gif := []byte("GIF89a-encoded-video")
	for _, mediaType := range []string{media.GIFMediaType, "video/mp4"} {
		handler, recipeID := videoOutputFixture(t, mediaType, gif)
		response := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations", `{"model":"`+recipeID.String()+`","prompt":"a fox on a trail"}`)
		if response.Code != http.StatusOK {
			t.Fatalf("%s: status=%d body=%s", mediaType, response.Code, response.Body.String())
		}
		var result nativeImageResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if len(result.Data) != 1 {
			t.Fatalf("%s: videos=%+v", mediaType, result.Data)
		}
		content := serveTestRequest(handler, http.MethodGet, result.Data[0].URL, "")
		if content.Code != http.StatusOK || content.Header().Get("Content-Type") != mediaType || !bytes.Equal(content.Body.Bytes(), gif) {
			t.Fatalf("%s: content status=%d type=%q", mediaType, content.Code, content.Header().Get("Content-Type"))
		}
	}
	handler, recipeID := videoOutputFixture(t, "text/plain", []byte("not a video"))
	refused := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations", `{"model":"`+recipeID.String()+`","prompt":"a fox on a trail"}`)
	if refused.Code != http.StatusInternalServerError || !bytes.Contains(refused.Body.Bytes(), []byte("invalid_output")) {
		t.Fatalf("text output status=%d body=%s", refused.Code, refused.Body.String())
	}
	if _, dedicated := routesByPath["/v1/videos/edits"]; dedicated {
		t.Fatal("the retired video-edit route is still routed")
	}
	if supported, _ := handler.workspaceCapability(t.Context(), "workflow.video-edit"); supported {
		t.Fatal("the retired video-edit workflow is still a workspace capability")
	}
}
