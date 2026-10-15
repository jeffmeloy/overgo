package main

import (
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"testing"
)

func TestReasoningAdapterUsesDeclaredClosure(t *testing.T) {
	for _, markers := range [][2]string{{"<think>", "</think>"}, {"<|channel>thought", "<channel|>"}} {
		t.Run(markers[0], func(t *testing.T) {
			path := testutil.HermeticLlamaGGUFWith(t, 64, gguf.StringMetadata("tokenizer.chat_template", markers[0]+markers[1]))
			loaded, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHost, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference)
			if err != nil {
				t.Fatal(err)
			}
			runner, err := inference.OpenWithProgram(t.Context(), &loaded, inference.OpenOptions{})
			if err != nil {
				_ = loaded.Close()
				t.Fatal(err)
			}
			defer runner.Close()
			runtime := chatShapedRuntime{runner}
			const tool = "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>"
			for _, tc := range []struct {
				name, text, want string
				closed           bool
			}{
				{"plain", "answer", "answer", false},
				{"tool_without_thought", "answer" + tool, "answer", false},
				{"closed_with_tool", markers[0] + "thought" + markers[1] + "answer" + tool, "answer", true},
				{"empty_thought", markers[0] + markers[1] + "answer", "answer", true},
				{"prompt_open", "thought" + markers[1] + "answer", "answer", true},
				{"unfinished", markers[0] + "thought", "", false},
				{"malformed_tool", markers[0] + markers[1] + "<tool_call>broken</tool_call>", "", false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					said, closed := runtime.AfterThought(tc.text)
					if said != tc.want || closed != tc.closed {
						t.Fatalf("got %q closed=%t, want %q closed=%t", said, closed, tc.want, tc.closed)
					}
				})
			}
		})
	}
}
