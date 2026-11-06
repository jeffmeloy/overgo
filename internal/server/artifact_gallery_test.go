package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestArtifactGalleryProjectedPageIsBoundedAndStreamsPayloads(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(overgodb.Create(root))(t)
	payload := []byte("image-payload")
	payloadID := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, payload))(t)
	payloadDescriptor := artifact.Descriptor{
		ID: payloadID, Size: uint64(len(payload)), MediaType: "image/png",
	}
	missingData := []byte("missing-payload")
	missingID := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, missingData))(t)
	missingDescriptor := artifact.Descriptor{ID: missingID, Size: uint64(len(missingData))}
	runData := []byte("producer-run")
	runID := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, runData))(t)
	runDescriptor := artifact.Descriptor{ID: runID, Size: uint64(len(runData))}
	testcheck.Must(artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:       "server/artifact-gallery",
		Artifacts: []artifact.Descriptor{missingDescriptor, runDescriptor},
		Contents:  []artifact.Content{{Descriptor: payloadDescriptor, Data: payload}},
		Lineage: []artifact.Lineage{
			{Child: payloadID, Parent: runID, Relation: artifact.RelationProducedBy},
			{Child: missingID, Parent: runID, Relation: artifact.RelationProducedBy},
		},
	}))(t)
	testcheck.NoError(t, store.Close())
	handler := testcheck.Must(New(Config{OvergoDBPath: root, MaxStoredResponses: 1}, &fakeGenerator{}))(t)
	t.Cleanup(func() { _ = handler.Close() })

	page := serveTestRequest(handler, http.MethodGet, "/artifacts?kind=output&limit=1", "")
	var pageResult artifactGalleryResponse
	testcheck.NoError(t, json.Unmarshal(page.Body.Bytes(), &pageResult))
	if page.Code != http.StatusOK || len(pageResult.Artifacts) != 1 || !pageResult.Truncated {
		t.Fatalf("page status=%d result=%+v", page.Code, pageResult)
	}
	if pageResult.Next == "" || pageResult.Count != 2 {
		t.Fatalf("page cursor/count = (%q, %d)", pageResult.Next, pageResult.Count)
	}
	nextPage := serveTestRequest(handler, http.MethodGet, "/artifacts?kind=output&limit=1&cursor="+pageResult.Next, "")
	var nextResult artifactGalleryResponse
	testcheck.NoError(t, json.Unmarshal(nextPage.Body.Bytes(), &nextResult))
	if nextPage.Code != http.StatusOK || len(nextResult.Artifacts) != 1 || nextResult.Next != "" ||
		nextResult.Artifacts[0].Descriptor.ID == pageResult.Artifacts[0].Descriptor.ID {
		t.Fatalf("next page status=%d result=%+v", nextPage.Code, nextResult)
	}

	available := serveTestRequest(handler, http.MethodGet, "/artifacts?id="+payloadID.String(), "")
	var availableResult artifactGalleryResponse
	testcheck.NoError(t, json.Unmarshal(available.Body.Bytes(), &availableResult))
	if len(availableResult.Artifacts) != 1 || !availableResult.Artifacts[0].Payload ||
		len(availableResult.Artifacts[0].Producers) != 1 || availableResult.Artifacts[0].Producers[0] != runID {
		t.Fatalf("available artifact = %+v", availableResult)
	}
	missing := serveTestRequest(handler, http.MethodGet, "/artifacts?id="+missingID.String(), "")
	var missingResult artifactGalleryResponse
	testcheck.NoError(t, json.Unmarshal(missing.Body.Bytes(), &missingResult))
	if len(missingResult.Artifacts) != 1 || missingResult.Artifacts[0].Payload {
		t.Fatalf("missing artifact = %+v", missingResult)
	}
	content := serveTestRequest(handler, http.MethodGet, "/artifacts/content?id="+payloadID.String(), "")
	if content.Code != http.StatusOK || content.Header().Get("Content-Type") != payloadDescriptor.MediaType ||
		!bytes.Equal(content.Body.Bytes(), payload) {
		t.Fatalf("content status=%d type=%q body=%q", content.Code, content.Header().Get("Content-Type"), content.Body.Bytes())
	}
	browse := handler.browseRepository
	t.Cleanup(func() { _ = browse.Close() })
	if again, err := handler.browseStore(t.Context()); err != nil || again != browse {
		t.Fatalf("browse store reopened: (%p, %p, %v)", browse, again, err)
	}
	testcheck.NoError(t, handler.Close())
	if err := browse.Refresh(t.Context()); !errors.Is(err, overgodb.ErrClosed) {
		t.Fatalf("refresh after handler close = %v", err)
	}
}
