package testutil

import (
	"context"
	"encoding/json"
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
