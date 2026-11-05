package testutil

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testcheck"
)

// ParseID parses value as an artifact identity, ending the test when it is none.
func ParseID(t testing.TB, value string) artifact.ID {
	t.Helper()
	id, err := artifact.ParseID(value)
	testcheck.NoError(t, err)
	return id
}

// HubModelInfo serves the hub's information for model acme/tiny at revision rev0: one file named filename whose large
// file holds weights.
func HubModelInfo(filename string, weights []byte) http.HandlerFunc {
	digest := sha256.Sum256(weights)
	return func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": filename, "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": len(weights),
			}}},
		})
	}
}

// JSON is value encoded as JSON.
func JSON(t testing.TB, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	testcheck.NoError(t, err)
	return string(encoded)
}

// ReadTypedJSON decodes into target the typed content reader holds under id.
func ReadTypedJSON(t testing.TB, reader artifact.Reader, id artifact.ID, target any) {
	t.Helper()
	content, err := artifact.RequireTypedContent(t.Context(), reader, id)
	testcheck.NoError(t, err)
	testcheck.NoError(t, json.Unmarshal(content.Data, target))
}

// Closer closes what a test opened under ctx, for the test's cleanup, reporting a close that fails.
func Closer(t testing.TB, ctx context.Context, opened interface{ Close(context.Context) error }) func() {
	return func() {
		if err := opened.Close(ctx); err != nil {
			t.Error(err)
		}
	}
}

// Metadata is GGUF metadata of valueType holding data.
func Metadata(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: data}}
}
