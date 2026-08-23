package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestArtifactGalleryProjectedPageIsBoundedAndStreamsPayloads(t *testing.T) {
	root := t.TempDir()
	store, err := repodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("image-payload")
	payloadID, err := artifact.IdentifyBytes(artifact.KindOutput, payload)
	if err != nil {
		t.Fatal(err)
	}
	payloadDescriptor := artifact.Descriptor{
		ID: payloadID, Size: uint64(len(payload)), MediaType: "image/png",
	}
	missingData := []byte("missing-payload")
	missingID, err := artifact.IdentifyBytes(artifact.KindOutput, missingData)
	if err != nil {
		t.Fatal(err)
	}
	missingDescriptor := artifact.Descriptor{ID: missingID, Size: uint64(len(missingData))}
	runData := []byte("producer-run")
	runID, err := artifact.IdentifyBytes(artifact.KindRun, runData)
	if err != nil {
		t.Fatal(err)
	}
	runDescriptor := artifact.Descriptor{ID: runID, Size: uint64(len(runData))}
	_, err = artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key:       "server/artifact-gallery",
		Artifacts: []artifact.Descriptor{missingDescriptor, runDescriptor},
		Contents:  []artifact.Content{{Descriptor: payloadDescriptor, Data: payload}},
		Lineage: []artifact.Lineage{
			{Child: payloadID, Parent: runID, Relation: artifact.RelationProducedBy},
			{Child: missingID, Parent: runID, Relation: artifact.RelationProducedBy},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{RepoDBPath: root, MaxStoredResponses: 1}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })

	page := serveTestRequest(handler, http.MethodGet, "/artifacts?kind=output&limit=1", "")
	var pageResult artifactGalleryResponse
	if err := json.Unmarshal(page.Body.Bytes(), &pageResult); err != nil {
		t.Fatal(err)
	}
	if page.Code != http.StatusOK || len(pageResult.Artifacts) != 1 || !pageResult.Truncated {
		t.Fatalf("page status=%d result=%+v", page.Code, pageResult)
	}
	if pageResult.Next == "" || pageResult.Count != 2 {
		t.Fatalf("page cursor/count = (%q, %d)", pageResult.Next, pageResult.Count)
	}
	nextPage := serveTestRequest(handler, http.MethodGet, "/artifacts?kind=output&limit=1&cursor="+pageResult.Next, "")
	var nextResult artifactGalleryResponse
	if err := json.Unmarshal(nextPage.Body.Bytes(), &nextResult); err != nil {
		t.Fatal(err)
	}
	if nextPage.Code != http.StatusOK || len(nextResult.Artifacts) != 1 || nextResult.Next != "" ||
		nextResult.Artifacts[0].Descriptor.ID == pageResult.Artifacts[0].Descriptor.ID {
		t.Fatalf("next page status=%d result=%+v", nextPage.Code, nextResult)
	}

	available := serveTestRequest(handler, http.MethodGet, "/artifacts?id="+payloadID.String(), "")
	var availableResult artifactGalleryResponse
	if err := json.Unmarshal(available.Body.Bytes(), &availableResult); err != nil {
		t.Fatal(err)
	}
	if len(availableResult.Artifacts) != 1 || !availableResult.Artifacts[0].Payload ||
		len(availableResult.Artifacts[0].Producers) != 1 || availableResult.Artifacts[0].Producers[0] != runID {
		t.Fatalf("available artifact = %+v", availableResult)
	}
	missing := serveTestRequest(handler, http.MethodGet, "/artifacts?id="+missingID.String(), "")
	var missingResult artifactGalleryResponse
	if err := json.Unmarshal(missing.Body.Bytes(), &missingResult); err != nil {
		t.Fatal(err)
	}
	if len(missingResult.Artifacts) != 1 || missingResult.Artifacts[0].Payload {
		t.Fatalf("missing artifact = %+v", missingResult)
	}
	content := serveTestRequest(handler, http.MethodGet, "/artifacts/content?id="+payloadID.String(), "")
	if content.Code != http.StatusOK || content.Header().Get("Content-Type") != payloadDescriptor.MediaType ||
		!bytes.Equal(content.Body.Bytes(), payload) {
		t.Fatalf("content status=%d type=%q body=%q", content.Code, content.Header().Get("Content-Type"), content.Body.Bytes())
	}

	module := serveTestRequest(handler, http.MethodGet, "/mod/artifacts.js", "").Body.String()
	for _, token := range []string{"item.payload", `type.startsWith("image/")`, `type.startsWith("audio/")`, `type.startsWith("video/")`} {
		if !strings.Contains(module, token) {
			t.Errorf("artifact module missing %q", token)
		}
	}
	if strings.Contains(module, "base64") {
		t.Error("artifact module retains encoded payloads")
	}
}
