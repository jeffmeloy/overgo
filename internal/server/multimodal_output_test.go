package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestMultimodalOutputProjection pins the inline-rendering contract:
// an interaction's media artifacts project from the replay endpoint
// with their declared media types, and the artifact content route
// serves the bytes under that same type -- so the GUI renders an
// image as an image, never as an opaque identity.
func TestMultimodalOutputProjection(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	ctx := t.Context()
	description, described := handler.interactionDescription()
	if !described {
		t.Fatal("serving fixture lacks an interaction description")
	}
	pngBytes := []byte{0x89, 'P', 'N', 'G', 0, 1, 2, 3}
	mediaID, err := artifact.IdentifyBytes(artifact.KindOutput, pngBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "multimodal-output-fixture",
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{
				ID: mediaID, MediaType: "image/png", Schema: "overgo/raw-media/v1", Size: uint64(len(pngBytes)),
			},
			Data: pngBytes,
		}},
		Artifacts: []artifact.Descriptor{{ID: description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishInteraction(ctx, store, runrecord.Interaction{
		Response: "media-response",
		Recipe:   description.Identity.Recipe, Model: description.Identity.Model,
		Node: description.Interaction.Node, Media: []artifact.ID{mediaID},
	}, []runrecord.InteractionMessage{
		{Role: string(inference.ChatRoleUser), Content: "draw"},
		{Role: string(inference.ChatRoleAssistant), Content: "drawn"},
	}, ""); err != nil {
		t.Fatal(err)
	}
	replay := serveTestRequest(handler, http.MethodGet, "/interactions/replay?response=media-response", "")
	if replay.Code != http.StatusOK ||
		!strings.Contains(replay.Body.String(), mediaID.String()) ||
		!strings.Contains(replay.Body.String(), `"media_type":"image/png"`) {
		t.Fatalf("replay media status=%d body=%s", replay.Code, replay.Body.String())
	}
	content := serveTestRequest(handler, http.MethodGet, "/artifacts/content?id="+mediaID.String(), "")
	if content.Code != http.StatusOK || content.Header().Get("Content-Type") != "image/png" ||
		content.Body.String() != string(pngBytes) {
		t.Fatalf("content status=%d type=%q", content.Code, content.Header().Get("Content-Type"))
	}
}
