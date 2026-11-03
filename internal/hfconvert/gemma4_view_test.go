package hfconvert

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/safetensors"
	"overgo/internal/testskip"
)

// gemma4Sample is how many bytes of each end of a tensor are compared: its layout and its first and last rows, without
// reading twelve billion weights.
const gemma4Sample = 1 << 16

// gemma4Stored are tensors a Gemma 4 conversion writes as the directory stores them, by the name each has in the GGUF.
var gemma4Stored = map[string]string{
	"token_embd.weight":       "model.language_model.embed_tokens.weight",
	"output_norm.weight":      "model.language_model.norm.weight",
	"blk.0.attn_q.weight":     "model.language_model.layers.0.self_attn.q_proj.weight",
	"blk.0.ffn_down.weight":   "model.language_model.layers.0.mlp.down_proj.weight",
	"blk.47.attn_norm.weight": "model.language_model.layers.47.input_layernorm.weight",
}

// TestAGemma4DirectoryIsTheBytesItsConversionWrites: a Gemma 4 directory is served from its own files as its conversion
// would write it: the directory's own bytes for each tensor it writes as stored, at both ends, and a layer's scalar
// widened to F32 as the conversion widens it. The view and the conversion share one list of tensors, so these are the
// conversion's bytes, read without writing twenty-four gigabytes.
func TestAGemma4DirectoryIsTheBytesItsConversionWrites(t *testing.T) {
	t.Parallel()
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT for the Gemma 4 directory")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Fatal(err)
	}
	directory := roots.ResolveModelPath("gemma-4-12B-it")
	if !Viewed(directory) {
		t.Fatalf("%s is not served from its own files", directory)
	}
	viewed, err := View(directory, "")
	if err != nil {
		t.Fatal(err)
	}
	defer viewed.Close()
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if architecture, found := viewed.MetadataValue("general.architecture"); !found || architecture.Data != "gemma4" {
		t.Fatalf("the view's architecture is %v", architecture)
	}
	read := func(tensor gguf.TensorInfo, offset, size uint64) []byte {
		t.Helper()
		data := make([]byte, size)
		if err := viewed.ReadTensorRange(tensor, offset, data); err != nil {
			t.Fatal(err)
		}
		return data
	}
	for name, stored := range gemma4Stored {
		got, found := viewed.Tensor(name)
		want, held := source.Tensors[stored]
		if !found || !held || got.Size != uint64(want.Size()) {
			t.Fatalf("tensor %s is %+v in the view, and the directory holds %s %t", name, got, stored, held)
		}
		sample := min(uint64(gemma4Sample), got.Size)
		for _, offset := range []uint64{0, got.Size - sample} {
			original := make([]byte, sample)
			if _, err := want.ReadAt(original, int64(offset)); err != nil && err != io.EOF {
				t.Fatal(err)
			}
			if !bytes.Equal(read(got, offset, sample), original) {
				t.Fatalf("tensor %s differs from %s at offset %d", name, stored, offset)
			}
		}
	}
	scalar, found := viewed.Tensor("blk.0.layer_output_scale.weight")
	stored := source.Tensors["model.language_model.layers.0.layer_scalar"]
	original := make([]byte, stored.Size())
	if _, err := stored.ReadAt(original, 0); err != nil && err != io.EOF {
		t.Fatal(err)
	}
	widened := math.Float32frombits(uint32(binary.LittleEndian.Uint16(original)) << 16)
	if !found || scalar.Type != gguf.DTypeF32 || math.Float32frombits(binary.LittleEndian.Uint32(read(scalar, 0, 4))) != widened {
		t.Fatalf("the layer scalar is %+v, want %v widened to F32", scalar, widened)
	}
}
