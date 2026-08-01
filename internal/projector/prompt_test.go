package projector

import (
	"strings"
	"testing"

	"llamacpp2go/internal/tokenizer"
)

func TestQwen35ImagePromptText(t *testing.T) {
	prompt := Qwen35ImagePromptText("before", "after", 3, true)
	if strings.Count(prompt, Qwen3VLImagePad) != 3 ||
		!strings.HasPrefix(prompt, "<|im_start|>user\nbefore<|vision_start|>") ||
		!strings.HasSuffix(prompt, "after<|im_end|>\n<|im_start|>assistant\n<think>\n") {
		t.Fatalf("unexpected prompt %q", prompt)
	}
	withoutThinking := Qwen35ImagePromptText("", "Describe.", 1, false)
	if !strings.HasSuffix(withoutThinking, "<think>\n\n</think>\n\n") {
		t.Fatalf("unexpected no-thinking prompt %q", withoutThinking)
	}
}

func TestQwen35VideoPromptText(t *testing.T) {
	prompt := Qwen35VideoPromptText("", "Question", 2, 2, 24, true)
	want := "<|im_start|>user\n<|vision_start|>" +
		"<0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>" +
		"<0.1 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>" +
		"<|vision_end|>Question<|im_end|>\n<|im_start|>assistant\n<think>\n"
	if prompt != want {
		t.Fatalf("video prompt = %q, want %q", prompt, want)
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
	positions, err := Qwen3VLMultiAxisPositions(10, 2, 4, 4, 4, 2)
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

func TestQwen3VLMultiChunkPositions(t *testing.T) {
	positions, err := Qwen3VLMultiChunkPositions(12, []int{2, 8}, 2, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := [4][]uint32{
		{0, 1, 2, 2, 4, 5, 6, 7, 8, 8, 10, 11},
		{0, 1, 2, 2, 4, 5, 6, 7, 8, 8, 10, 11},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		{0, 1, 0, 0, 4, 5, 6, 7, 0, 0, 10, 11},
	}
	for axis := range positions {
		for index := range positions[axis] {
			if positions[axis][index] != want[axis][index] {
				t.Fatalf("axis %d position %d = %d, want %d", axis, index, positions[axis][index], want[axis][index])
			}
		}
	}
}
