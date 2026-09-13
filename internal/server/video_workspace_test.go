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
	"overgo/internal/testutil"
)

func videoWorkspaceFixture(t *testing.T) (*Handler, *nativeMediaWorkspace, []byte) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
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
				Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.video"}}},
				Controls: []WorkflowControl{{Name: "prompt", Type: WorkflowControlText, Required: true}},
			},
		},
		completion: map[recipe.Task]operation.Completion{
			recipe.TaskVideoGen: {Run: generateRun, Outputs: []artifact.ID{videoID}},
		},
	}
	handler, err := New(Config{Repository: store}, workspace)
	if err != nil {
		t.Fatal(err)
	}
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
	handler, workspace, video := videoWorkspaceFixture(t)
	recipeID := workspace.capabilities[0].Recipe
	response := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations",
		`{"model":"`+recipeID.String()+`","prompt":"a fox on a trail"}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result nativeImageResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
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
