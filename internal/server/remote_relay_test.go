package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// newRelayGenerator: relay over a fake provider answering pieces, its
// remote environment, the provider's received-request record.
func newRelayGenerator(t *testing.T, pieces []string) (*remoterelay.Generator, runrecord.Environment, *relaytest.Received) {
	t.Helper()
	provider, received := relaytest.Serve(t, "", "relay-key", pieces)
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
	generator, err := remoterelay.New(declared, definition, provider.Client())
	if err != nil {
		t.Fatal(err)
	}
	environment, err := remoteprovider.Environment(declared)
	if err != nil {
		t.Fatal(err)
	}
	return generator, environment, received
}

// TestResponsesRelayStoresInteraction: repository-backed handler stores a
// relayed response's interaction (relay description carries the relay
// node's interaction scope); the front page's remote turns depend on it.
func TestResponsesRelayStoresInteraction(t *testing.T) {
	generator, environment, _ := newRelayGenerator(t, []string{"Hello"})
	policy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = %v supported=%v", err, supported)
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	// The interaction's lineage names the served recipe: declared in the store, as a declaration commits it.
	description, err := generator.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Commit(t.Context(), artifact.Batch{
		Key: "test/serving-recipe/" + description.Identity.Recipe.String(), Artifacts: []artifact.Descriptor{{ID: description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		RuntimePolicy: policy, ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature,
		DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy, Environment: environment, Repository: repository,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	response := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"model":"`+testModelID+`","input":"hi","max_output_tokens":4}`)
	var stored struct {
		ID string `json:"id"`
	}
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &stored) != nil || stored.ID == "" {
		t.Fatalf("relayed response status=%d body=%s", response.Code, response.Body.String())
	}
	// The response's usage is the provider's accounting, not the relay's piece count.
	if !strings.Contains(response.Body.String(), `"input_tokens":7,"output_tokens":1,"total_tokens":8`) {
		t.Fatalf("relayed usage: %s", response.Body.String())
	}
	if _, found, err := runrecord.ResolveInteraction(t.Context(), repository, stored.ID); err != nil || !found {
		t.Fatalf("stored interaction for %s: found=%v err=%v", stored.ID, found, err)
	}
}

// TestChatCompletionsRelayToRemoteProvider pins the served relay: a chat
// completion over a handler whose generator is the remote relay reaches
// the provider's endpoint as the conversation, the streamed deltas come
// back as the protocol's chunks, the declared environment says the
// provider's runs are not reproducible, and the count route refuses (the
// provider tokenizes) rather than reporting zero tokens.
func TestChatCompletionsRelayToRemoteProvider(t *testing.T) {
	generator, environment, received := newRelayGenerator(t, []string{"Hel", "lo"})
	policy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = %v supported=%v", err, supported)
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
	// Capabilities document declares remote serving -> page marks it.
	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if manifest.Code != http.StatusOK || !strings.Contains(manifest.Body.String(), `"remote":true`) {
		t.Fatalf("manifest status=%d body=%s", manifest.Code, manifest.Body.String())
	}
}

// TestFrontPageRemoteTurns: page remote marks (picker entry, welcome card,
// assistant turns) + turn proceeds when the count route refuses (provider
// tokenizes).
func TestFrontPageRemoteTurns(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	if boot := get("/boot.js"); !strings.Contains(boot, `startsWith("remote://")`) || !strings.Contains(boot, `text: "remote"`) {
		t.Error("the picker does not mark remote entries")
	}
	if composer := get("/composer.js"); !strings.Contains(composer, "options.marker") {
		t.Error("the thread renders no turn marker")
	}
	chat := get("/mod/chat.js")
	for _, needle := range []string{`marker: capabilities.remote`, `capabilities.remote ? ["remote"] : []`, `.catch(() => null)`, `count ? count.input_tokens : null`, `usage.prompt_tokens`} {
		if !strings.Contains(chat, needle) {
			t.Errorf("the chat page lacks %q", needle)
		}
	}
}
