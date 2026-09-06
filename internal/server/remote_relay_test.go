package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/testutil"
)

// TestChatCompletionsRelayToRemoteProvider pins the served relay: a chat
// completion over a handler whose generator is the remote relay reaches
// the provider's endpoint as the conversation, the streamed deltas come
// back as the protocol's chunks, the declared environment says the
// provider's runs are not reproducible, and the count route refuses (the
// provider tokenizes) rather than reporting zero tokens.
func TestChatCompletionsRelayToRemoteProvider(t *testing.T) {
	provider, received := relaytest.Serve(t, "", "relay-key", []string{"Hel", "lo"})
	t.Setenv("OVERGO_SERVER_RELAY_TEST_KEY", "relay-key")
	declared, err := remoteprovider.New(remoteprovider.Provider{
		Name: "fake", Endpoint: provider.URL, KeyEnvironment: "OVERGO_SERVER_RELAY_TEST_KEY", Model: "vendor/model",
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.RemoteInferenceDefinition(testutil.ArtifactID(t, artifact.KindModel, "served-remote"), declared.ID)
	if err != nil {
		t.Fatal(err)
	}
	policy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = %v supported=%v", err, supported)
	}
	generator, err := remoterelay.New(declared, definition, provider.Client())
	if err != nil {
		t.Fatal(err)
	}
	environment, err := remoteprovider.Environment(declared)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		RuntimePolicy: policy, ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature,
		DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy, Environment: environment,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()

	body := `{"model":"` + testModelID + `","max_tokens":4,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`
	response := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", body)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"content":"Hello"`) {
		t.Fatalf("relayed completion status=%d body=%s", response.Code, response.Body.String())
	}
	if received.Model != "vendor/model" || len(received.Messages) != 2 || received.Messages[1].Content != "hi" {
		t.Fatalf("provider received %+v", *received)
	}
	streamed := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", strings.TrimSuffix(body, "}")+`,"stream":true}`)
	if streamed.Code != http.StatusOK || !strings.Contains(streamed.Body.String(), `"content":"Hel"`) || !strings.Contains(streamed.Body.String(), "[DONE]") {
		t.Fatalf("streamed completion status=%d body=%s", streamed.Code, streamed.Body.String())
	}
	if environment.Reproducible() {
		t.Fatal("a remote environment reads as reproducible")
	}
	count := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions/input_tokens", body)
	if count.Code != http.StatusNotImplemented || !strings.Contains(count.Body.String(), "token counting is unavailable") {
		t.Fatalf("relayed token count status=%d body=%s", count.Code, count.Body.String())
	}
}
