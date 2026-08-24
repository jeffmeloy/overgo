package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type nativeMediaWorkspace struct {
	*fakeGenerator
	capabilities []WorkflowCapability
	completion   map[recipe.Task]operation.Completion
	mu           sync.Mutex
	executed     []recipe.Task
}

func (workspace *nativeMediaWorkspace) WorkflowCapabilities(context.Context, WorkflowKind) ([]WorkflowCapability, error) {
	return workspace.capabilities, nil
}

func (workspace *nativeMediaWorkspace) ExecuteWorkflow(
	_ context.Context,
	_ WorkflowKind,
	task recipe.Task,
	_ artifact.ID,
	_ json.RawMessage,
	reporter operation.Reporter,
) (operation.Completion, error) {
	workspace.mu.Lock()
	workspace.executed = append(workspace.executed, task)
	workspace.mu.Unlock()
	reporter.Publishing()
	return workspace.completion[task], nil
}

func nativeMediaProtocolFixture(t *testing.T) (*Handler, *nativeMediaWorkspace, []byte, []byte) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	image, audio := []byte("image-output"), tinyPCM16WAV()
	imageID := testutil.ArtifactBytesID(t, artifact.KindOutput, image)
	audioID := testutil.ArtifactBytesID(t, artifact.KindOutput, audio)
	imageRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "native-image-recipe")
	audioRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "native-audio-recipe")
	imageRun := testutil.ArtifactID(t, artifact.KindRun, "native-image-run")
	audioRun := testutil.ArtifactID(t, artifact.KindRun, "native-audio-run")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "native/media/outputs",
		Artifacts: []artifact.Descriptor{
			{ID: imageRecipe}, {ID: audioRecipe}, {ID: imageRun}, {ID: audioRun},
		},
		Contents: []artifact.Content{
			{Descriptor: artifact.Descriptor{ID: imageID, Size: uint64(len(image)), MediaType: "image/png"}, Data: image},
			{Descriptor: artifact.Descriptor{ID: audioID, Size: uint64(len(audio)), MediaType: "audio/wav"}, Data: audio},
		},
	}); err != nil {
		t.Fatal(err)
	}
	workspace := &nativeMediaWorkspace{
		fakeGenerator: &fakeGenerator{},
		capabilities: []WorkflowCapability{
			{
				Task: recipe.TaskImageGen, Recipe: imageRecipe,
				Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.image"}}},
				Controls: []WorkflowControl{{Name: "prompt", Type: WorkflowControlText, Required: true}},
			},
			{
				Task: recipe.TaskSpeech, Recipe: audioRecipe,
				Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.audio"}}},
				Controls: []WorkflowControl{{Name: "input", Type: WorkflowControlText, Required: true}},
			},
		},
		completion: map[recipe.Task]operation.Completion{
			recipe.TaskImageGen: {Run: imageRun, Outputs: []artifact.ID{imageID}},
			recipe.TaskSpeech:   {Run: audioRun, Outputs: []artifact.ID{audioID}},
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
	return handler, workspace, image, audio
}

func TestNativeImageProtocolUsesActiveRecipeSession(t *testing.T) {
	handler, workspace, image, _ := nativeMediaProtocolFixture(t)
	recipeID := workspace.capabilities[0].Recipe
	response := serveTestRequest(handler, http.MethodPost, "/v1/images/generations",
		`{"model":"`+recipeID.String()+`","prompt":"test","response_format":"url"}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result nativeImageResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Data) != 1 {
		t.Fatalf("images=%+v", result.Data)
	}
	content := serveTestRequest(handler, http.MethodGet, result.Data[0].URL, "")
	if content.Code != http.StatusOK || !bytes.Equal(content.Body.Bytes(), image) {
		t.Fatalf("content status=%d body=%q", content.Code, content.Body.Bytes())
	}
	if len(workspace.executed) != 1 || workspace.executed[0] != recipe.TaskImageGen {
		t.Fatalf("executed=%v", workspace.executed)
	}
}

func TestNativeAudioProtocolUsesActiveRecipeSession(t *testing.T) {
	handler, workspace, _, audio := nativeMediaProtocolFixture(t)
	recipeID := workspace.capabilities[1].Recipe
	response := serveTestRequest(handler, http.MethodPost, "/v1/audio/speech",
		`{"model":"`+recipeID.String()+`","input":"test"}`)
	if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "audio/wav" ||
		!bytes.Equal(response.Body.Bytes(), audio) {
		t.Fatalf("status=%d type=%q body=%q", response.Code, response.Header().Get("Content-Type"), response.Body.Bytes())
	}
	if len(workspace.executed) != 1 || workspace.executed[0] != recipe.TaskSpeech {
		t.Fatalf("executed=%v", workspace.executed)
	}
}
