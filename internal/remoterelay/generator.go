// Package remoterelay serves a remote model: the generator the HTTP
// server runs when the served model is a declared provider's, carrying
// each chat turn to the provider's OpenAI-compatible chat completions
// endpoint and streaming the answer back as token events. The relay
// forwards text only: the messages the server assembled, the decode
// budget and the stop sequences; the provider samples under its own
// policy, and the answer's tokens are the pieces it streamed.
package remoterelay

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/runrecord"
	"overgo/internal/tokenizer"
)

// completionsPath is the chat completions path under the provider's
// endpoint, the OpenAI protocol's.
const completionsPath = "/chat/completions"

// Generator relays generation to one declared provider.
type Generator struct {
	provider   remoteprovider.Provider
	key        string
	definition recipe.Definition
	client     *http.Client
}

// Conversation is the prompt form the relay's chat formatter emits and its
// generator decodes: the messages themselves, so the provider receives the
// conversation rather than a template flattened for a local model.
type Conversation struct {
	Messages []inference.ChatMessage `json:"messages"`
}

// New binds the relay to a provider whose key the environment holds; a
// provider without its key is refused by name. A nil client uses the
// default one.
func New(provider remoteprovider.Provider, definition recipe.Definition, client *http.Client) (*Generator, error) {
	key, err := remoteprovider.Key(provider)
	if err != nil {
		return nil, err
	}
	return &Generator{provider: provider, key: key, definition: definition, client: cmp.Or(client, http.DefaultClient)}, nil
}

// FormatChat renders the conversation the generator forwards; media parts
// are refused since the relay forwards text only.
func (g *Generator) FormatChat(messages []inference.ChatMessage) (string, error) {
	for _, message := range messages {
		if len(message.Media) != 0 {
			return "", errors.New("remote relay: media parts are not forwarded to a remote provider")
		}
	}
	data, err := json.Marshal(Conversation{Messages: messages})
	if err != nil {
		return "", err
	}
	return string(data), nil
}

type wireMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitzero"`
}

type completionRequest struct {
	Model     string        `json:"model"`
	Messages  []wireMessage `json:"messages"`
	Stream    bool          `json:"stream"`
	MaxTokens int           `json:"max_tokens,omitzero"`
	Stop      []string      `json:"stop,omitempty"`
	// StreamOptions asks the provider to close the stream with its usage
	// chunk (the OpenAI stream protocol's include_usage).
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// completionUsage is the provider's token accounting for the completion.
type completionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type completionChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		// FinishReason closes the choice ("stop", "length"): a terminal marker beside [DONE].
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *completionUsage `json:"usage"`
	// Error is the provider's error event inside the stream; it ends the answer.
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// conversationOf decodes the formatter's conversation, or wraps a plain
// prompt (the completions route's) as one user message.
func conversationOf(prompt string) []wireMessage {
	var conversation Conversation
	if err := json.Unmarshal([]byte(prompt), &conversation); err == nil && len(conversation.Messages) != 0 {
		messages := make([]wireMessage, 0, len(conversation.Messages))
		for _, message := range conversation.Messages {
			messages = append(messages, wireMessage{Role: string(message.Role), Content: message.Content, Name: message.Name})
		}
		return messages
	}
	return []wireMessage{{Role: string(inference.ChatRoleUser), Content: prompt}}
}

// Generate posts the conversation to the provider and streams the answer:
// every content delta is one token event and one piece of the text. The
// provider's stop ends the stream; the caller's stop predicate ends it
// early. No token ids exist for a remote answer.
func (g *Generator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	body, err := json.Marshal(completionRequest{
		Model: g.provider.Model, Messages: conversationOf(prompt), Stream: true,
		MaxTokens: options.MaxNewTokens, Stop: options.StopSequences,
		StreamOptions: &streamOptions{IncludeUsage: true},
	})
	if err != nil {
		return nil, "", err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, g.provider.Endpoint+completionsPath, bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	request.Header.Set("Authorization", "Bearer "+g.key)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	response, err := g.client.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return nil, "", ctx.Err()
		}
		return nil, "", fmt.Errorf("remote relay: %s: %w", g.provider.Name, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, bufio.MaxScanTokenSize))
		return nil, "", fmt.Errorf("remote relay: %s answered %s: %s", g.provider.Name, response.Status, strings.TrimSpace(string(detail)))
	}
	var text strings.Builder
	index := 0
	// completed: the stream reached its terminal marker ([DONE] or a finish reason); an EOF
	// before it is a truncated answer, never a success.
	completed := false
	scanner := bufio.NewScanner(response.Body)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			completed = true
			break
		}
		var chunk completionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, text.String(), fmt.Errorf("remote relay: decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return nil, text.String(), fmt.Errorf("remote relay: %s reported %s: %s", g.provider.Name, cmp.Or(chunk.Error.Type, "an error"), chunk.Error.Message)
		}
		// The provider's accounting closes the stream: it tokenizes, so its counts are the turn's.
		if chunk.Usage != nil && options.OnUsage != nil {
			options.OnUsage(inference.Usage{PromptTokens: chunk.Usage.PromptTokens, CompletionTokens: chunk.Usage.CompletionTokens})
		}
		for _, choice := range chunk.Choices {
			completed = completed || choice.FinishReason != ""
			piece := choice.Delta.Content
			if piece == "" {
				continue
			}
			text.WriteString(piece)
			event := inference.TokenEvent{Piece: piece, Index: index}
			index++
			if options.OnToken != nil {
				if err := options.OnToken(event); err != nil {
					return nil, text.String(), err
				}
			}
			if options.ShouldStop != nil && options.ShouldStop(event) {
				return nil, text.String(), nil
			}
		}
	}
	if err := scanner.Err(); err != nil {
		// The caller's cancellation is its own error, not a provider failure.
		if ctx.Err() != nil {
			return nil, text.String(), ctx.Err()
		}
		return nil, text.String(), fmt.Errorf("remote relay: read stream: %w", err)
	}
	if !completed {
		return nil, text.String(), fmt.Errorf("remote relay: %s ended the stream without its terminal marker after %d piece(s)", g.provider.Name, index)
	}
	return nil, text.String(), nil
}

// ModelProperties describes the remote model from its declaration: the
// remote location as its path, the provider's model id as its name, the
// remote backend as its architecture and the declared context length.
func (g *Generator) ModelProperties() inference.ModelProperties {
	return inference.ModelProperties{
		Path: remoteprovider.Location(g.provider), Name: g.provider.Model,
		Architecture: runrecord.BackendRemote, ContextLength: g.provider.ContextLength,
	}
}

// RecipeRuntimeDescription binds served recipe + model + the relay node's
// interaction scope, so observations and stored interactions attach to the
// remote activation.
func (g *Generator) RecipeRuntimeDescription(task recipe.Task) (modelrecipe.RuntimeDescription, error) {
	if task != recipe.TaskInference {
		return modelrecipe.RuntimeDescription{}, fmt.Errorf("remote relay: active %s recipe is unavailable", task)
	}
	profile, _ := g.definition.PrimaryDependency(recipe.DependencyProfile)
	return modelrecipe.DescribeDefinition(modelrecipe.ProgramIdentity{
		Model: g.definition.Model, Profile: profile, Definition: g.definition.ID, Recipe: g.definition.ID,
		Placement: recipe.PlacementHost,
	}, g.definition, nil)
}
