package hfgguf

import (
	"bytes"
	"io"
	"slices"
	"testing"

	"overgo/internal/safetensors"
)

func TestQwen35ProjectorMappingAndTemporalSplit(t *testing.T) {
	for source, want := range map[string]string{
		"model.visual.pos_embed.weight":              "v.position_embd.weight",
		"model.visual.blocks.7.attn.qkv.weight":      "v.blk.7.attn_qkv.weight",
		"model.visual.blocks.23.mlp.linear_fc2.bias": "v.blk.23.ffn_down.bias",
		"model.visual.merger.linear_fc2.weight":      "mm.2.weight",
	} {
		if got, ok := qwen35ProjectorTensorName(source); !ok || got != want {
			t.Fatalf("map %q = %q/%v, want %q", source, got, ok, want)
		}
	}
	config := qwen35VisionConfig{HiddenSize: 2, PatchSize: 1, TemporalPatchSize: 2}
	raw := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	}
	tensor, err := safetensors.NewTensor(
		"model.visual.patch_embed.proj.weight", "BF16", []uint64{2, 3, 2, 1, 1},
		bytes.NewReader(raw), 0, int64(len(raw)),
	)
	if err != nil {
		t.Fatal(err)
	}
	parts, err := qwen35TemporalPatchTensors(tensor, config)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range [][]byte{
		{0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21},
		{2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23},
	} {
		got, err := io.ReadAll(parts[index].Data)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("temporal %d = %v, want %v", index, got, want)
		}
	}
}

func TestResolveQwen35VisionRope(t *testing.T) {
	if got, err := resolveQwen35VisionRope(nil); err != nil || got != qwen35VisionDefaultRope {
		t.Fatalf("default rope = %v, %v", got, err)
	}
	explicit := float32(25000)
	if got, err := resolveQwen35VisionRope(&explicit); err != nil || got != explicit {
		t.Fatalf("explicit rope = %v, %v", got, err)
	}
	invalid := float32(0)
	if _, err := resolveQwen35VisionRope(&invalid); err == nil {
		t.Fatal("invalid explicit rope accepted")
	}
}
