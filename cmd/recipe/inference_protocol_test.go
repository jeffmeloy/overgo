package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// protocolAnswers are what the protocol fixture answers a prompt holding each
// question of fixtures/protocol_acceptance.json; the tool request answers a
// call the fixture's parser reads.
var protocolAnswers = []struct{ question, answer string }{
	{"17 plus 25", "42"},
	{"amber shipment code", "ORBIT-731"},
	{"What is my code", "719"},
	{"7 plus 5", "12"},
	{"lookup_code", "call lookup_code amber"},
	{"3 plus 4", "7"},
	{"Count from 1 through 20", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
	{"5 plus 4", "9"},
}

// protocolFixture is a served model off the device: it answers each
// protocol case's question a piece per token, swapping in wrong for the
// answer of the first, and records whether a generation was cancelled and
// whether it was closed.
type protocolFixture struct {
	wrong     string
	cancelled bool
	closed    bool
}

// RuntimePolicy is the catalog's inference policy, which the handlers serve under.
func (*protocolFixture) RuntimePolicy() modelrecipe.RuntimePolicy {
	policy, _, _ := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	return policy
}

// FormatChat joins the conversation's contents.
func (fixture *protocolFixture) FormatChat(messages []inference.ChatMessage) (string, error) {
	return fixture.FormatChatWithOptions(messages, inference.ChatFormatOptions{})
}

// FormatChatWithOptions joins the conversation's contents and the names of its tools.
func (*protocolFixture) FormatChatWithOptions(messages []inference.ChatMessage, options inference.ChatFormatOptions) (string, error) {
	var prompt strings.Builder
	for _, message := range messages {
		prompt.WriteString(message.Content + "\n")
	}
	for _, tool := range options.Tools {
		prompt.WriteString(tool.Function.Name + "\n")
	}
	return prompt.String(), nil
}

// ParseChatOutput reads a call the fixture answered as a tool call.
func (*protocolFixture) ParseChatOutput(output string, _ []inference.ChatTool) (inference.ChatMessage, error) {
	fields := strings.Fields(output)
	if len(fields) != 3 || fields[0] != "call" {
		return inference.ChatMessage{Role: inference.ChatRoleAssistant, Content: output}, nil
	}
	return inference.ChatMessage{Role: inference.ChatRoleAssistant, ToolCalls: []inference.ChatToolCall{{
		Type: inference.ChatToolTypeFunction, Function: inference.ChatToolFunction{Name: fields[1], Arguments: `{"color":"` + fields[2] + `"}`},
	}}}, nil
}

// ChatToolGrammar constrains nothing: the fixture answers its call unconstrained.
func (*protocolFixture) ChatToolGrammar([]inference.ChatTool, bool, bool, bool) (string, string, []string, error) {
	return "", "", nil, nil
}

// Generate answers the question the prompt holds a piece per token.
func (fixture *protocolFixture) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	answer := ""
	for index, known := range protocolAnswers {
		if strings.Contains(prompt, known.question) {
			answer = known.answer
			if index == 0 && fixture.wrong != "" {
				answer = fixture.wrong
			}
			break
		}
	}
	// The prompt's ids lead the answer's, a word a token, as a runner returns them.
	ids := make([]tokenizer.TokenID, len(strings.Fields(prompt)))
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(ids)})
	}
	for index, piece := range strings.SplitAfter(answer, " ") {
		if options.OnToken != nil {
			if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(index), Piece: piece, Index: index}); err != nil {
				fixture.cancelled = errors.Is(err, context.Canceled)
				return ids, "", err
			}
		}
		ids = append(ids, tokenizer.TokenID(index))
	}
	return ids, answer, ctx.Err()
}

// Close records the release.
func (fixture *protocolFixture) Close() error {
	fixture.closed = true
	return nil
}

// TestVerifyPublishesProtocolAcceptance holds the protocol route of recipe
// verify to its publication: the cases of fixtures/protocol_acceptance.json
// go through the serving handlers, and when the served model answers every
// one, cancellation included, the gate and run are published with the run
// binding the observation; when one case is answered wrongly nothing is.
func TestVerifyPublishesProtocolAcceptance(t *testing.T) {
	document, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "protocol_acceptance.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, wrong := range []string{"", "43"} {
		t.Run("wrong="+wrong, func(t *testing.T) {
			ctx := t.Context()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
				t.Fatal(err)
			}
			candidate, err := prepareExactCandidate(ctx, store, testutil.HermeticLlamaGGUF(t, servedContext), modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
			if err != nil {
				t.Fatal(err)
			}
			head, sequence := store.Head()
			model := &protocolFixture{wrong: wrong}
			verification, err := acceptProtocol(ctx, store, candidate.Definition, candidate.Resolved.Document.ID, strings.Repeat("a", 40), string(document),
				func(context.Context) (exactRuntime, error) { return model, nil })
			if !model.closed {
				t.Fatal("the served model was not closed")
			}
			if wrong != "" {
				if err == nil || !strings.Contains(err.Error(), "chat-arithmetic") || verification != (modelrecipe.Verification{}) {
					t.Fatalf("a wrong answer gave %+v, %v; want the arithmetic case refused", verification, err)
				}
				if now, ordinal := store.Head(); now != head || ordinal != sequence {
					t.Fatal("a refused protocol acceptance published to the store")
				}
				return
			}
			if err != nil || !model.cancelled {
				t.Fatalf("protocol acceptance = %v, cancelled generation %t; want every case accepted", err, model.cancelled)
			}
			gate, err := runrecord.RequireGateResult(ctx, store, verification.Gate)
			if err != nil || gate.Outcome != runrecord.OutcomeSucceeded || gate.Recipe != candidate.Definition.ID {
				t.Fatalf("gate %+v, %v; want a succeeded gate of the candidate", gate, err)
			}
			run, err := runrecord.RequireRun(ctx, store, verification.Run)
			if err != nil || len(run.Inputs) != 1 {
				t.Fatalf("run %+v, %v; want the run bound to the observation", run, err)
			}
			observation, found, err := artifact.ReadContent(ctx, store, run.Inputs[0])
			var cases []protocolCase
			if err != nil || !found || strictjson.DecodeBytes(observation.Data, &cases) != nil || len(cases) != len(protocolAnswers) {
				t.Fatalf("observation %s found %t, %v, %d cases; want the %d cases", run.Inputs[0], found, err, len(cases), len(protocolAnswers))
			}
			for _, test := range cases {
				if !test.Accepted {
					t.Fatalf("observed case %+v unaccepted", test)
				}
			}
			// The gate and run are what recipe activate takes.
			if err := modelrecipe.ActivateCapability(ctx, store, candidate.Definition, verification, recipe.EvidenceVerified, "protocol cases accepted"); err != nil {
				t.Fatal(err)
			}
			if active, found, err := modelrecipe.ActiveRecord(ctx, store, candidate.Definition.Model, recipe.TaskInference); err != nil || !found || active.Definition.ID != candidate.Definition.ID {
				t.Fatalf("active recipe %+v found %t, %v; want the protocol-accepted candidate", active.Definition.ID, found, err)
			}
			// The gate names the exact plan its first chat case recorded, the request recipe measure replays.
			active, requests, err := verifiedCase(ctx, store, candidate.Definition.Model)
			if err != nil || active != candidate.Definition.ID || len(requests) != 1 || requests[0].Name != "chat-arithmetic" ||
				!strings.Contains(requests[0].Prompt, "17 plus 25") || requests[0].MaxTokens != 16 || requests[0].Text != "42" {
				t.Fatalf("recipe measure replays %+v of %s (%v); want the arithmetic chat case as the candidate answered it", requests, active, err)
			}
		})
	}
}
