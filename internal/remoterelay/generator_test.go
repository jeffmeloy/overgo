package remoterelay

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
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
	var usage inference.Usage
	_, text, err := generator.Generate(t.Context(), prompt, inference.GenerateOptions{
		MaxNewTokens: 7, StopSequences: []string{"END"},
		OnToken: func(event inference.TokenEvent) error {
			if event.Index != len(pieces) {
				t.Fatalf("token index %d, want %d", event.Index, len(pieces))
			}
			pieces = append(pieces, event.Piece)
			return nil
		},
		OnUsage: func(reported inference.Usage) { usage = reported },
	})
	if err != nil || text != "Hello world" || strings.Join(pieces, "|") != "Hel|lo| world" {
		t.Fatalf("relay = %q pieces=%v err=%v", text, pieces, err)
	}
	// The relay asks for the usage chunk and hands the provider's counts on.
	if received.Model != "fake/model" || !received.Stream || received.MaxTokens != 7 || len(received.Stop) != 1 || !received.StreamOptions.IncludeUsage ||
		len(received.Messages) != 2 || received.Messages[0].Role != "system" || received.Messages[1].Content != "hello?" {
		t.Fatalf("provider received %+v", *received)
	}
	if usage != (inference.Usage{PromptTokens: relaytest.UsagePromptTokens, CompletionTokens: 3}) {
		t.Fatalf("usage = %+v", usage)
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

// TestGeneratorRefusesTruncatedCancelledAndErroredStreams pins the stream's
// ending: an EOF before the terminal marker is a truncated answer with the
// text so far, a finish reason closes a stream without [DONE], the
// provider's in-stream error event is the error, and the caller's
// cancellation comes back as its own error once the pieces have arrived.
func TestGeneratorRefusesTruncatedCancelledAndErroredStreams(t *testing.T) {
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "test-key")
	relay := func(t *testing.T, behaviour relaytest.Behaviour) *Generator {
		t.Helper()
		server, _ := relaytest.ServeBehaviour(t, "/api/v1", "test-key", []string{"par", "tial"}, nil, behaviour)
		provider, definition := testProvider(t, server.URL)
		generator, err := New(provider, definition, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		return generator
	}
	t.Run("truncated", func(t *testing.T) {
		_, text, err := relay(t, relaytest.Behaviour{Truncate: true}).Generate(t.Context(), "hello", inference.GenerateOptions{})
		if err == nil || !strings.Contains(err.Error(), "without its terminal marker after 2 piece(s)") || text != "partial" {
			t.Fatalf("truncated stream = %q, %v", text, err)
		}
	})
	t.Run("provider error event", func(t *testing.T) {
		_, text, err := relay(t, relaytest.Behaviour{ErrorEvent: "overloaded"}).Generate(t.Context(), "hello", inference.GenerateOptions{})
		if err == nil || !strings.Contains(err.Error(), "fake reported server_error: overloaded") || text != "partial" {
			t.Fatalf("errored stream = %q, %v", text, err)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		// The caller leaves once the second piece is in hand; the held stream then ends with its error.
		_, text, err := relay(t, relaytest.Behaviour{Hold: true}).Generate(ctx, "hello", inference.GenerateOptions{
			OnToken: func(event inference.TokenEvent) error {
				if event.Index == 1 {
					cancel(errors.New("the caller left"))
				}
				return nil
			},
		})
		if !errors.Is(err, context.Canceled) || text != "partial" {
			t.Fatalf("cancelled stream = %q, %v", text, err)
		}
	})
	t.Run("finish reason closes the stream", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			response.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(response, "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n")
		}))
		defer server.Close()
		provider, definition := testProvider(t, server.URL)
		generator, err := New(provider, definition, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		if _, text, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{}); err != nil || text != "done" {
			t.Fatalf("finished stream = %q, %v", text, err)
		}
	})
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

func TestGeneratorUsesUpdatedKey(t *testing.T) {
	server, _ := relaytest.Serve(t, "/api/v1", "updated-key", []string{"updated"})
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "old-key")
	provider, definition := testProvider(t, server.URL)
	generator, err := New(provider, definition, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{}); err == nil {
		t.Fatal("old key unexpectedly accepted")
	}
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "updated-key")
	if _, answer, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{}); err != nil || answer != "updated" {
		t.Fatalf("active relay ignored replacement key: answer=%q err=%v", answer, err)
	}
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "")
	if _, _, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{}); err == nil || !strings.Contains(err.Error(), "is not set") {
		t.Fatalf("withdrawn key remained active: %v", err)
	}
}
