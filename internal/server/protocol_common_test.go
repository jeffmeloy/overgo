package server

import (
	"errors"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

type testToolOutputStream struct {
	deltas []inference.ChatToolCallDelta
}

func (s *testToolOutputStream) Accept(string) ([]inference.ChatToolCallDelta, error) {
	return s.deltas, nil
}

func (*testToolOutputStream) Finish() (inference.ChatMessage, error) {
	return inference.ChatMessage{}, errors.New("unused")
}

func TestBoundedProtocolTokens(t *testing.T) {
	defaultTokens := testRuntimePolicy().Serving.OutputTokens
	value := 4
	invalid := 33
	for _, test := range []struct {
		name       string
		configured *int
		required   bool
		want       int
		wantError  string
	}{
		{name: "default", want: defaultTokens},
		{name: "required", required: true, wantError: "max_tokens is required"},
		{name: "configured", configured: &value, want: 4},
		{name: "bounded", configured: &invalid, wantError: "must be in [0,32]"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := boundedProtocolTokens(
				test.configured, defaultTokens, 32, "max_tokens", test.required,
			)
			if test.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("error = %v", err)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("tokens = %d, error = %v", got, err)
			}
		})
	}
}

func TestProtocolGenerationOptionsPinsProjectedPrompt(t *testing.T) {
	handler := &Handler{config: Config{ContextShift: true}}
	projected := &inference.ProjectedInputs{}
	options := handler.protocolGenerationOptions(
		3, &sampling.Sampler{}, []tokenizer.TokenID{1, 2}, projected,
	)
	if options.MaxNewTokens != 3 || options.Sampler == nil || !options.ParseSpecial ||
		!options.ContextShift || !options.CachePrompt || options.ProjectedInputs != projected ||
		len(options.PromptTokenIDs) != 2 {
		t.Fatalf("options = %+v", options)
	}
}

func TestToolDeltaStreamOwnsBufferAndCallState(t *testing.T) {
	stream := &toolDeltaStream{output: &testToolOutputStream{deltas: []inference.ChatToolCallDelta{{
		Index: 1, Name: "lookup", Arguments: `{"x":`, Started: true,
	}}}}
	if _, err := stream.accept("first"); err != nil {
		t.Fatal(err)
	}
	stream.output = &testToolOutputStream{deltas: []inference.ChatToolCallDelta{{
		Index: 1, Arguments: "1}",
	}}}
	if _, err := stream.accept("second"); err != nil {
		t.Fatal(err)
	}
	if stream.text() != "firstsecond" || !stream.started || !stream.streamed(1) ||
		stream.argumentText(1) != `{"x":1}` {
		t.Fatalf("stream = %+v", stream)
	}
}

func TestToolDeltaStreamRejectsNegativeIndex(t *testing.T) {
	stream := &toolDeltaStream{output: &testToolOutputStream{deltas: []inference.ChatToolCallDelta{{
		Index: -1, Started: true,
	}}}}
	if _, err := stream.accept("bad"); err == nil {
		t.Fatal("negative tool-stream index accepted")
	}
}
