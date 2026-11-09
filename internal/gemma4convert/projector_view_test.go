// Package gemma4convert_test holds the Gemma 4 converter to what it serves through the packages that open models: a
// projector opened from a checkpoint's own directory is the one its conversion writes.
package gemma4convert_test

import (
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gemma4convert"
	_ "overgo/internal/hfconvert"
	"overgo/internal/projector"
	"overgo/internal/testcheck"
)

// projectorViewConfig is a Gemma 4 checkpoint of the embedder layout gemma-4-12B-it has (no vision_config
// num_hidden_layers), at a size a test reads: hidden 8, a patch of 2 pooled by 2, 16 positions, audio embedding 4.
const projectorViewConfig = `{
  "model_type": "gemma4_unified",
  "text_config": {"hidden_size": 8, "intermediate_size": 8, "num_hidden_layers": 1, "layer_types": ["full_attention"],
    "num_attention_heads": 1, "head_dim": 8, "global_head_dim": 8, "num_key_value_heads": 1, "vocab_size": 4,
    "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
    "rope_parameters": {"full_attention": {"rope_type": "proportional", "rope_theta": 1000000, "partial_rotary_factor": 0.25},
      "sliding_attention": {"rope_type": "default", "rope_theta": 10000}}},
  "vision_config": {"mm_embed_dim": 8, "mm_posemb_size": 16, "patch_size": 2, "pooling_kernel_size": 2, "rms_norm_eps": 1e-6},
  "audio_config": {"audio_embed_dim": 4, "rms_norm_eps": 1e-6}
}`

// projectorViewProcessor is the checkpoint's processor configuration: the token budgets and sample rate the
// projector's metadata states.
const projectorViewProcessor = `{"feature_extractor": {"sampling_rate": 16000},
  "image_processor": {"max_soft_tokens": 4}, "video_processor": {"max_soft_tokens": 4}}`

// projectorViewShapes are the checkpoint's projector tensors, in the checkpoint's own row-major shapes.
var projectorViewShapes = map[string][]uint64{
	"model.embed_vision.patch_dense.weight":                              {8, 48},
	"model.embed_vision.patch_dense.bias":                                {8},
	"model.embed_vision.patch_ln1.weight":                                {48},
	"model.embed_vision.patch_ln1.bias":                                  {48},
	"model.embed_vision.patch_ln2.weight":                                {8},
	"model.embed_vision.patch_ln2.bias":                                  {8},
	"model.embed_vision.pos_embedding":                                   {16, 2, 8},
	"model.embed_vision.pos_norm.weight":                                 {8},
	"model.embed_vision.pos_norm.bias":                                   {8},
	"model.embed_vision.multimodal_embedder.embedding_projection.weight": {8, 8},
	"model.embed_audio.embedding_projection.weight":                      {8, 4},
}

// TestAProjectorIsReadFromItsPublishedFiles: a Gemma 4 checkpoint's media projector opens from the checkpoint's own
// directory, as its text does, with nothing written there, and is the projector its conversion writes: the same spec,
// and the same embeddings of a picture to the bit. gemma-4-12B-it's vision was reachable only through a converted copy,
// which the rule that a model is served from its tensors as published forbids (judge-calibration/published-tower).
func TestAProjectorIsReadFromItsPublishedFiles(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(t.TempDir(), "gemma-4-fixture")
	testcheck.NoError(t, os.MkdirAll(directory, 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), []byte(projectorViewConfig), 0o644))
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "processor_config.json"), []byte(projectorViewProcessor), 0o644))
	writeBF16Safetensors(t, filepath.Join(directory, "model.safetensors"), projectorViewShapes)
	published := testcheck.Must(os.ReadDir(directory))(t)

	converted := filepath.Join(t.TempDir(), "mmproj.gguf")
	testcheck.Must(gemma4convert.Convert(gemma4convert.Options{Directory: directory, MMProjPath: converted}))(t)
	fromFile := testcheck.Must(projector.OpenAs[*projector.Gemma4Runner](t.Context(), converted, projector.OpenOptions{}))(t)
	defer fromFile.Close()
	fromDirectory := testcheck.Must(projector.OpenAs[*projector.Gemma4Runner](t.Context(), directory, projector.OpenOptions{}))(t)
	defer fromDirectory.Close()
	if fromFile.Spec() != fromDirectory.Spec() {
		t.Fatalf("the projector read from the directory has spec %+v, the converted one %+v", fromDirectory.Spec(), fromFile.Spec())
	}
	picture := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for index := range picture.Pix {
		picture.Pix[index] = uint8(index * 7)
	}
	picture.Set(0, 0, color.RGBA{R: 255, A: 255})
	want := testcheck.Must(fromFile.EncodeImage(t.Context(), picture))(t)
	got := testcheck.Must(fromDirectory.EncodeImage(t.Context(), picture))(t)
	if len(want.Embeddings.Data) == 0 || !slices.ContainsFunc(want.Embeddings.Data, func(value float32) bool { return value != 0 }) {
		t.Fatalf("the converted projector embedded the picture as %v, want values", want.Embeddings.Data)
	}
	if !slices.Equal(got.Embeddings.Data, want.Embeddings.Data) || got.GridH != want.GridH || got.GridW != want.GridW {
		t.Fatalf("the projector read from the directory embedded the picture as %v (%d x %d), the converted one as %v (%d x %d)",
			got.Embeddings.Data, got.GridH, got.GridW, want.Embeddings.Data, want.GridH, want.GridW)
	}
	if after := testcheck.Must(os.ReadDir(directory))(t); len(after) != len(published) {
		t.Fatalf("reading the projector from the directory left %d entries in it, want the %d published", len(after), len(published))
	}
}

// writeBF16Safetensors writes a safetensors file of BF16 tensors of the shapes given, their values drawn from a fixed
// seed in [-1, 1).
func writeBF16Safetensors(t *testing.T, path string, shapes map[string][]uint64) {
	t.Helper()
	const bf16Bytes = 2
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	slices.Sort(names)
	source := rand.New(rand.NewPCG(uint64(len(names)), bf16Bytes))
	header := map[string]any{}
	var data []byte
	for _, name := range names {
		count := uint64(1)
		for _, extent := range shapes[name] {
			count *= extent
		}
		start := len(data)
		for range count {
			bits := math.Float32bits(source.Float32()*2 - 1)
			data = binary.LittleEndian.AppendUint16(data, uint16(bits>>16))
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []int{start, len(data)}}
	}
	encoded := testcheck.Must(json.Marshal(header))(t)
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(encoded)))
	testcheck.NoError(t, os.WriteFile(path, append(append(file, encoded...), data...), 0o644))
}
