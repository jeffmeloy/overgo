package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestArtifactGalleryListsNewestFirstByMedia pins the slot strip's listing:
// newest first by the commit that introduced each file, kept to the media
// prefix asked for, bounded by the limit, with no cursor.
func TestArtifactGalleryListsNewestFirstByMedia(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	commit := func(key, mediaType string, data []byte) artifact.ID {
		t.Helper()
		id, err := artifact.IdentifyBytes(artifact.KindFile, data)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
			Key:      key,
			Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: mediaType}, Data: data}},
		}); err != nil {
			t.Fatal(err)
		}
		return id
	}
	older := commit("server/gallery-newest/1", "image/png", []byte("older image"))
	clip := commit("server/gallery-newest/2", "audio/wav", []byte("a clip"))
	newer := commit("server/gallery-newest/3", "image/jpeg", []byte("newer image"))
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{OvergoDBPath: root, MaxStoredResponses: 8}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	list := func(query string) []artifact.ID {
		t.Helper()
		page := serveTestRequest(handler, http.MethodGet, "/artifacts?"+query, "")
		var result artifactGalleryResponse
		if page.Code != http.StatusOK || json.Unmarshal(page.Body.Bytes(), &result) != nil || result.Next != "" {
			t.Fatalf("%s: status=%d body=%s", query, page.Code, page.Body.String())
		}
		ids := make([]artifact.ID, 0, len(result.Artifacts))
		for _, item := range result.Artifacts {
			if !item.Payload {
				t.Fatalf("%s listed %s without its payload", query, item.Descriptor.ID)
			}
			ids = append(ids, item.Descriptor.ID)
		}
		return ids
	}
	if ids := list("kind=file&newest=1&media=image/&limit=1"); len(ids) != 1 || ids[0] != newer {
		t.Fatalf("newest image = %v, want %s", ids, newer)
	}
	if ids := list("kind=file&newest=1&media=image/&limit=5"); len(ids) != 2 || ids[0] != newer || ids[1] != older {
		t.Fatalf("images newest first = %v", ids)
	}
	if ids := list("kind=file&newest=1&media=audio/&limit=5"); len(ids) != 1 || ids[0] != clip {
		t.Fatalf("audio = %v, want %s", ids, clip)
	}
	if ids := list("kind=file&newest=1&limit=5"); len(ids) != 3 || ids[0] != newer || ids[2] != older {
		t.Fatalf("every file newest first = %v", ids)
	}
}
