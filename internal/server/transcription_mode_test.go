package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipecontract"
)

// TestTranscriptionModeRunsThroughGenericRoute: the front page's
// transcription mode. The capability document declares the mode enabled;
// the capability lists an artifact-typed audio control; a WAV stored
// through the intake route fills it; the generic run route completes with
// the run's transcription document and the transcript's text beside it,
// the text under the shared text answer contract.
func TestTranscriptionModeRunsThroughGenericRoute(t *testing.T) {
	fixture := newTranscriptionHTTPFixture(t, nil)
	authorized := func(method, path, body, mediaType string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(method, path, strings.NewReader(body))
		request.Header.Set("Authorization", "Bearer "+testAPIKey)
		if mediaType != "" {
			request.Header.Set("Content-Type", mediaType)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	if manifest := authorized(http.MethodGet, "/workspace/manifest", "", ""); manifest.Code != http.StatusOK ||
		!strings.Contains(manifest.Body.String(), `{"id":"transcription","label":"Transcribe","enabled":true}`) {
		t.Fatalf("manifest status=%d body=%s", manifest.Code, manifest.Body.String())
	}
	if capabilities := authorized(http.MethodGet, "/generation/capabilities", "", ""); capabilities.Code != http.StatusOK ||
		!strings.Contains(capabilities.Body.String(), `"controls":[{"name":"audio","type":"artifact","required":true,"label":"audio clip","media":"audio"}]`) {
		t.Fatalf("capabilities status=%d body=%s", capabilities.Code, capabilities.Body.String())
	}
	var stored attachmentIntake
	if intake := authorized(http.MethodPost, "/artifacts/intake", string(fixture.wave), "audio/wav"); intake.Code != http.StatusOK ||
		json.Unmarshal(intake.Body.Bytes(), &stored) != nil {
		t.Fatalf("intake status=%d body=%s", intake.Code, intake.Body.String())
	}
	accepted := authorized(http.MethodPost, "/generation/run",
		`{"task":"transcription","recipe":"`+fixture.workspace.policy.Recipe.String()+`","input":{"audio":"`+stored.ID+`"}}`, "application/json")
	var submission workflowResponse
	if accepted.Code != http.StatusAccepted || json.Unmarshal(accepted.Body.Bytes(), &submission) != nil {
		t.Fatalf("run status=%d body=%s", accepted.Code, accepted.Body.String())
	}
	waited := authorized(http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "", "")
	var status operation.Status
	if waited.Code != http.StatusOK || json.Unmarshal(waited.Body.Bytes(), &status) != nil || status.State != operation.StateCompleted || len(status.Outputs) != 2 {
		t.Fatalf("wait status=%d body=%s", waited.Code, waited.Body.String())
	}
	text := authorized(http.MethodGet, "/artifacts/content?id="+status.Outputs[1].String(), "", "")
	if text.Code != http.StatusOK || text.Header().Get("Content-Type") != modelrecipe.TextAnswerMediaType || text.Body.String() != "x" {
		t.Fatalf("text output status=%d type=%s body=%q", text.Code, text.Header().Get("Content-Type"), text.Body.String())
	}
	var transcription recipecontract.Transcription
	if document := authorized(http.MethodGet, "/artifacts/content?id="+status.Outputs[0].String(), "", ""); document.Code != http.StatusOK ||
		json.Unmarshal(document.Body.Bytes(), &transcription) != nil || transcription.Text != text.Body.String() {
		t.Fatalf("transcription document status=%d body=%s", document.Code, document.Body.String())
	}
}
