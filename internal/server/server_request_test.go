package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func serveTestRequest(
	handler http.Handler,
	method, path, body string,
) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func marshalTestJSON(t *testing.T, value any) string {
	t.Helper()
	return testutil.JSON(t, value)
}

func commitTestBlob(t *testing.T, store artifact.Repository, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	data := []byte(label)
	id, err := artifact.IdentifyBytes(kind, data)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "server/test/blob/" + label,
		Contents: []artifact.Content{{Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream",
		}, Data: data}},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}
