package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestWriteJSONRefusesAnUnencodableAnswer keeps an answer that cannot be
// encoded from reaching the client as an empty success: the active recipe
// once answered 200 with no body when its identity held a zero artifact ID,
// and the recipe tab failed reading a null answer.
func TestWriteJSONRefusesAnUnencodableAnswer(t *testing.T) {
	t.Parallel()
	response := httptest.NewRecorder()
	writeJSON(response, http.StatusOK, struct {
		ID artifact.ID `json:"id"`
	}{})
	var envelope apiErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("body %q: %v", response.Body.String(), err)
	}
	if response.Code != http.StatusInternalServerError || envelope.Error.Type != "encoding_error" || !strings.Contains(envelope.Error.Message, "invalid ID") {
		t.Fatalf("unencodable answer = %d %+v", response.Code, envelope)
	}
	encoded := httptest.NewRecorder()
	writeJSON(encoded, http.StatusCreated, map[string]int{"count": 1})
	if encoded.Code != http.StatusCreated || encoded.Body.String() != "{\"count\":1}\n" {
		t.Fatalf("encodable answer = %d %q", encoded.Code, encoded.Body.String())
	}
}
