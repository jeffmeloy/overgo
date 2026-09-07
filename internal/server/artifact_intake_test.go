package server

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// intakeRequest: POST body under its media type through handler.
func intakeRequest(handler http.Handler, mediaType, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, "/artifacts/intake", strings.NewReader(body))
	request.Header.Set("Content-Type", mediaType)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

// TestArtifactIntakeStoresAcceptedFiles: a decodable file stores as a file
// document of its media type (idempotent; the id passes the artifact
// control's validation) whatever the served chat model's projectors; an
// image is decoded against the image bounds; an undecodable type, an
// oversized body, an empty one and a handler without a repository refuse.
func TestArtifactIntakeStoresAcceptedFiles(t *testing.T) {
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, &fakeGenerator{})
	response := intakeRequest(handler, "text/plain; charset=utf-8", "an attached note")
	var stored attachmentIntake
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &stored) != nil {
		t.Fatalf("intake status=%d body=%s", response.Code, response.Body.String())
	}
	id, err := artifact.ParseID(stored.ID)
	if err != nil || id.Kind() != artifact.KindFile || stored.MediaType != "text/plain" || stored.Size != 16 {
		t.Fatalf("stored = %+v, %v", stored, err)
	}
	content, found, err := artifact.ReadContent(t.Context(), repository, id)
	if err != nil || !found || string(content.Data) != "an attached note" || content.Descriptor.MediaType != "text/plain" || content.Descriptor.Schema != attachmentSchema {
		t.Fatalf("stored content = %+v found=%v err=%v", content.Descriptor, found, err)
	}
	if err := validateWorkflowValue(WorkflowControlArtifact, json.RawMessage(`"`+stored.ID+`"`)); err != nil {
		t.Fatalf("the stored id fails the artifact control: %v", err)
	}
	if again := intakeRequest(handler, "text/plain", "an attached note"); again.Code != http.StatusOK || !strings.Contains(again.Body.String(), stored.ID) {
		t.Fatalf("second intake status=%d body=%s", again.Code, again.Body.String())
	}
	png, err := base64.StdEncoding.DecodeString(tinyPNGBase64(t))
	if err != nil {
		t.Fatal(err)
	}
	if image := intakeRequest(handler, "image/png", string(png)); image.Code != http.StatusOK || !strings.Contains(image.Body.String(), `"media_type":"image/png"`) {
		t.Fatalf("image intake status=%d body=%s", image.Code, image.Body.String())
	}
	if broken := intakeRequest(handler, "image/png", "not a png"); broken.Code != http.StatusBadRequest {
		t.Fatalf("undecodable image intake status=%d body=%s", broken.Code, broken.Body.String())
	}
	if refused := intakeRequest(handler, "application/zip", "PK"); refused.Code != http.StatusUnsupportedMediaType ||
		!strings.Contains(refused.Body.String(), "does not decode application/zip") {
		t.Fatalf("archive intake status=%d body=%s", refused.Code, refused.Body.String())
	}
	if oversized := intakeRequest(handler, "text/plain", strings.Repeat("x", maxMediaBytes+1)); oversized.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized intake status=%d body=%s", oversized.Code, oversized.Body.String())
	}
	if empty := intakeRequest(handler, "text/plain", ""); empty.Code != http.StatusBadRequest {
		t.Fatalf("empty intake status=%d body=%s", empty.Code, empty.Body.String())
	}
	if unavailable := intakeRequest(newTestHandler(t, &fakeGenerator{}), "text/plain", "note"); unavailable.Code != http.StatusServiceUnavailable {
		t.Fatalf("intake without a repository status=%d body=%s", unavailable.Code, unavailable.Body.String())
	}
}
