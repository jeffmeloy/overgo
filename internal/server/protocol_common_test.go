package server

import (
	"errors"
	"strings"
	"testing"

	"overgo/internal/inference"
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
	t.Parallel()
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

func TestToolDeltaStreamRejectsNegativeIndex(t *testing.T) {
	t.Parallel()
	stream := &toolDeltaStream{output: &testToolOutputStream{deltas: []inference.ChatToolCallDelta{{
		Index: -1, Started: true,
	}}}}
	if _, err := stream.accept("bad"); err == nil {
		t.Fatal("negative tool-stream index accepted")
	}
}
