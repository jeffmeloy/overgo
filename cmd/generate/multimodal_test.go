package main

import (
	"strings"
	"testing"

	"llamacpp2go/internal/tokenizer"
)

func TestQwen3VLPromptText(t *testing.T) {
	prompt := qwen3VLPromptText("Describe.", 3, true)
	if strings.Count(prompt, qwen3VLImagePad) != 3 ||
		!strings.HasPrefix(prompt, "<|im_start|>user\n<|vision_start|>") ||
		!strings.HasSuffix(prompt, "<|im_start|>assistant\n<think>\n") {
		t.Fatalf("unexpected prompt %q", prompt)
	}
	withoutThinking := qwen3VLPromptText("Describe.", 1, false)
	if !strings.HasSuffix(withoutThinking, "<think>\n\n</think>\n\n") {
		t.Fatalf("unexpected no-thinking prompt %q", withoutThinking)
	}
}

func TestContiguousTokenRun(t *testing.T) {
	ids := []tokenizer.TokenID{1, 7, 7, 7, 2}
	start, err := contiguousTokenRun(ids, 7, 3)
	if err != nil || start != 1 {
		t.Fatalf("run = %d, %v", start, err)
	}
	if _, err := contiguousTokenRun(ids, 7, 2); err == nil {
		t.Fatal("short placeholder count accepted")
	}
}

func TestQwen3VLMultiAxisPositions(t *testing.T) {
	positions, err := qwen3VLMultiAxisPositions(10, 2, 4, 4, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := [4][]uint32{
		{0, 1, 2, 2, 2, 2, 4, 5, 6, 7},
		{0, 1, 2, 2, 3, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 2, 3, 4, 5, 6, 7},
		{0, 1, 0, 0, 0, 0, 4, 5, 6, 7},
	}
	for axis := range positions {
		for index := range positions[axis] {
			if positions[axis][index] != want[axis][index] {
				t.Fatalf("axis %d position %d = %d, want %d", axis, index, positions[axis][index], want[axis][index])
			}
		}
	}
}
