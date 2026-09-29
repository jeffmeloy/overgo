package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
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
	capability WorkflowCapability,
	_ json.RawMessage,
	reporter operation.Reporter,
) (operation.Completion, error) {
	workspace.mu.Lock()
	workspace.executed = append(workspace.executed, capability.Task)
	workspace.mu.Unlock()
	reporter.Publishing()
	return workspace.completion[capability.Task], nil
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
	if _, err := store.Commit(t.Context(), artifact.Batch{
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
				Task: recipe.TaskImageGen, Recipe: imageRecipe, Outputs: []recipe.Output{{Name: "image", Data: recipe.DataImage}},
				Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.image"}}},
				Controls: []workflowControl{{Name: "prompt", Type: workflowControlText, Required: true}},
			},
			{
				Task: recipe.TaskSpeech, Recipe: audioRecipe, Outputs: []recipe.Output{{Name: "audio", Data: recipe.DataAudio}},
				Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.audio"}}},
				Controls: []workflowControl{{Name: "input", Type: workflowControlText, Required: true}},
			},
		},
		completion: map[recipe.Task]operation.Completion{
			recipe.TaskImageGen: {Run: imageRun, Outputs: []artifact.ID{imageID}},
			recipe.TaskSpeech:   {Run: audioRun, Outputs: []artifact.ID{audioID}},
		},
	}
	handler, err := New(Config{Repository: store, Workflows: workspace}, workspace)
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
	t.Parallel()
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
	t.Parallel()
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

// TestMediaRequestResolvesItsCapabilityOnce: a native image request served
// the way cmd/server serves it, the store's generation workspace in a
// workspace set, lists the capabilities once and runs the capability as that
// listing resolved it. A listing resolves each capability and declares its
// controls once, so the declarations count the listings.
func TestMediaRequestResolvesItsCapabilityOnce(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	activatedImageModel(t, store)
	var listings, executions atomic.Int32
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
			executions.Add(1)
			return tinyPNG(t), nil
		},
	}}
	controls := func(entry recipe.ModuleID, directory string) ([]mediacapability.Control, string) {
		listings.Add(1)
		return mediacapability.Controls(entry, directory)
	}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, controls, mediacapability.OutputContent), 16)
	handler, err := New(Config{Repository: store, Workflows: WorkflowWorkspaceSet{workspace}}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	response := serveTestRequest(handler, http.MethodPost, "/v1/images/generations", `{"class":1,"seed":7}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if listings.Load() != 1 || executions.Load() != 1 {
		t.Fatalf("one image request listed the capabilities %d times and executed %d times", listings.Load(), executions.Load())
	}
}
