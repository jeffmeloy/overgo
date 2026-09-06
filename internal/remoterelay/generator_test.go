package remoterelay

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/testutil"
)

func testProvider(t *testing.T, endpoint string) (remoteprovider.Provider, recipe.Definition) {
	t.Helper()
	provider, err := remoteprovider.New(remoteprovider.Provider{
		Name: "fake", Endpoint: endpoint + "/api/v1/", KeyEnvironment: "OVERGO_REMOTE_RELAY_TEST_KEY", Model: "fake/model", ContextLength: 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "remote-relay-model")
	definition, err := modelrecipe.RemoteInferenceDefinition(modelID, provider.ID)
	if err != nil {
		t.Fatal(err)
	}
	return provider, definition
}

// TestGeneratorRelaysTheConversationAndStreams pins the relay's wire form:
// the formatter's conversation reaches the provider as its messages with
// the decode budget and the stop sequences, the streamed deltas arrive as
// token events in order, the text is their concatenation, and the model
// properties describe the declaration.
func TestGeneratorRelaysTheConversationAndStreams(t *testing.T) {
	server, received := relaytest.Serve(t, "/api/v1", "test-key", []string{"Hel", "lo", " world"})
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "test-key")
	provider, definition := testProvider(t, server.URL)
	generator, err := New(provider, definition, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := generator.FormatChat([]inference.ChatMessage{
		{Role: inference.ChatRoleSystem, Content: "be brief"},
		{Role: inference.ChatRoleUser, Content: "hello?"},
	})
	if err != nil {
		t.Fatal(err)
	}
	var pieces []string
	_, text, err := generator.Generate(t.Context(), prompt, inference.GenerateOptions{
		MaxNewTokens: 7, StopSequences: []string{"END"},
		OnToken: func(event inference.TokenEvent) error {
			if event.Index != len(pieces) {
				t.Fatalf("token index %d, want %d", event.Index, len(pieces))
			}
			pieces = append(pieces, event.Piece)
			return nil
		},
	})
	if err != nil || text != "Hello world" || strings.Join(pieces, "|") != "Hel|lo| world" {
		t.Fatalf("relay = %q pieces=%v err=%v", text, pieces, err)
	}
	if received.Model != "fake/model" || !received.Stream || received.MaxTokens != 7 || len(received.Stop) != 1 ||
		len(received.Messages) != 2 || received.Messages[0].Role != "system" || received.Messages[1].Content != "hello?" {
		t.Fatalf("provider received %+v", *received)
	}
	properties := generator.ModelProperties()
	if properties.Name != "fake/model" || properties.Path != "remote://fake/fake/model" || properties.ContextLength != 4096 {
		t.Fatalf("properties = %+v", properties)
	}
	// Interaction scope valid: the server stores each response's interaction under it.
	description, err := generator.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil || description.Identity.Model != definition.Model || description.Identity.Recipe != definition.ID ||
		!description.Interaction.Valid() || description.Interaction.Node != "relay" {
		t.Fatalf("description = %+v, %v", description, err)
	}
	// A plain prompt (the completions route's) travels as one user message.
	if _, text, err := generator.Generate(t.Context(), "plain", inference.GenerateOptions{}); err != nil || text != "Hello world" ||
		len(received.Messages) != 1 || received.Messages[0].Role != "user" || received.Messages[0].Content != "plain" {
		t.Fatalf("plain prompt relay = %q %+v err=%v", text, received.Messages, err)
	}
}

// TestGeneratorRefusesWithoutKeyAndReportsProviderErrors pins the two
// refusals: a provider whose key the environment lacks is refused by the
// variable's name before any request, and a provider's error answer is
// reported with its status and body.
func TestGeneratorRefusesWithoutKeyAndReportsProviderErrors(t *testing.T) {
	server, _ := relaytest.Serve(t, "/api/v1", "test-key", nil)
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "")
	provider, definition := testProvider(t, server.URL)
	if _, err := New(provider, definition, server.Client()); err == nil ||
		!strings.Contains(err.Error(), "OVERGO_REMOTE_RELAY_TEST_KEY is not set") {
		t.Fatalf("keyless relay error = %v", err)
	}
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "wrong-key")
	generator, err := New(provider, definition, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{}); err == nil ||
		!strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "missing key") {
		t.Fatalf("provider error = %v", err)
	}
	if _, err := generator.FormatChat([]inference.ChatMessage{{Role: inference.ChatRoleUser, Media: []inference.ChatMediaPart{{}}}}); err == nil {
		t.Fatal("a media part was forwarded")
	}
}
