package projector

import (
	"context"
	"image"
	"image/color"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

type deepSeekOCRPromptTokenizer struct{ texts []string }

func (t *deepSeekOCRPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	t.texts = append(t.texts, text)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		if strings.HasPrefix(text, DeepSeekOCRImagePad) {
			ids, text = append(ids, 8), text[len(DeepSeekOCRImagePad):]
		} else {
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestDeepSeekOCRTinyFixture(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCRRunner](writeTinyDeepSeekOCR(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 32, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(3, 3)) || !slices.Equal(output.Data, make([]float32, 9)) {
		t.Fatalf("output = shape %v values %v", output.Shape, output.Data)
	}
}

func TestDeepSeekOCRRequiresPreprocessingMetadata(t *testing.T) {
	for _, missing := range []string{
		"clip.vision.preproc_image_size", "clip.vision.preproc_min_tiles", "clip.vision.preproc_max_tiles",
	} {
		metadata := slices.DeleteFunc(slices.Clone(tinyDeepSeekOCRMetadata()), func(item gguf.Metadata) bool {
			return item.Key == missing
		})
		path := writeProjectorFixture(t, "deepseekocr-missing-preprocessing.gguf", metadata, tinyDeepSeekOCRTensors(false))
		if _, err := openImageProjectorAs[*DeepSeekOCRRunner](path, OpenOptions{}); err == nil {
			t.Fatalf("missing metadata %q accepted", missing)
		}
	}
}

func TestDeepSeekOCRPreprocessesLocalTilesBeforeOverview(t *testing.T) {
	spec := tinyDeepSeekOCRSpec()
	input, err := PreprocessDeepSeekOCRImage(image.NewRGBA(image.Rect(0, 0, 128, 64)), spec)
	if err != nil {
		t.Fatal(err)
	}
	if input.GridW != 2 || input.GridH != 1 || len(input.Tiles) != 3 {
		t.Fatalf("grid = %dx%d, tiles = %d", input.GridW, input.GridH, len(input.Tiles))
	}
	if got := input.Tiles[len(input.Tiles)-1].Bounds().Size(); got != (image.Point{X: 64, Y: 64}) {
		t.Fatalf("overview size = %v", got)
	}
}

func TestDeepSeekOCRCanDisableDynamicTiles(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCRRunner](
		writeTinyDeepSeekOCR(t, false), OpenOptions{DisableDynamicTiles: true})

	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 128, 64)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(mustShape(3, 3)) {
		t.Fatalf("output shape = %v", output.Shape)
	}
}

func TestDeepSeekOCRPromptContract(t *testing.T) {
	runner, err := openImageProjectorAs[*DeepSeekOCRRunner](writeTinyDeepSeekOCR(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tok := &deepSeekOCRPromptTokenizer{}
	prompt, err := testSession(t, runner).BuildImagePrompt(context.Background(), tok, image.NewRGBA(image.Rect(0, 0, 32, 32)), "before", "after", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tok.texts) < 2 || tok.texts[0] != "before"+strings.Repeat(DeepSeekOCRImagePad, 3)+"after" {
		t.Fatalf("prompt text = %q", tok.texts)
	}
	if prompt.EmbeddingWidth != 3 || len(prompt.EmbeddingTokenIndices) != 3 || len(prompt.Embeddings) != 9 {
		t.Fatalf("prompt = width %d indices %d embeddings %d", prompt.EmbeddingWidth, len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings))
	}
}

func TestOpenImageProjectorDispatchesDeepSeekOCR(t *testing.T) {
	projector, err := OpenAs[Projector](context.Background(), writeTinyDeepSeekOCR(t, false), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*DeepSeekOCRRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestDeepSeekOCRCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyDeepSeekOCR(t, true)
	cpu, err := openImageProjectorAs[*DeepSeekOCRRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*DeepSeekOCRRunner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 7), G: uint8(y * 5), B: uint8((x + y) * 3), A: fixtureOpaqueAlpha})
		}
	}
	want, err := cpu.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "DeepSeek-OCR", got.Data, want.Data, 4e-3)
}

func writeTinyDeepSeekOCR(t *testing.T, patterned bool) string {
	return writeProjectorFixture(t, "deepseekocr-mmproj.gguf", tinyDeepSeekOCRMetadata(), tinyDeepSeekOCRTensors(patterned))
}

func tinyDeepSeekOCRSpec() DeepSeekOCRSpec {
	return DeepSeekOCRSpec{ImageSize: 64, TileSize: 64, MinTiles: 2, MaxTiles: 9, PatchSize: 16,
		Hidden: 2, FeedForward: 3, Layers: 1, Heads: 1, SAMHidden: 2, SAMLayers: 1, SAMHeads: 1,
		Window: 2, OutputHidden: 3, LayerNormEpsilon: 1e-5, ImageStd: [3]float32{1, 1, 1}, SAMGlobalLayers: []bool{false}}
}

func tinyDeepSeekOCRMetadata() []gguf.Metadata {
	spec := tinyDeepSeekOCRSpec()
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: deepSeekOCRProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.ImageSize)}},
		{Key: "clip.vision.preproc_image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.TileSize)}},
		{Key: "clip.vision.preproc_min_tiles", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.MinTiles)}},
		{Key: "clip.vision.preproc_max_tiles", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.MaxTiles)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.PatchSize)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.Hidden)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.FeedForward)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.Layers)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.Heads)}},
		{Key: "clip.vision.sam.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.SAMHidden)}},
		{Key: "clip.vision.sam.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.SAMLayers)}},
		{Key: "clip.vision.sam.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.SAMHeads)}},
		{Key: "clip.vision.sam.global_attention_layers", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: spec.SAMGlobalLayers}},
		{Key: "clip.vision.window_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(spec.Window)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: spec.LayerNormEpsilon}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyDeepSeekOCRTensors(patterned bool) []gguf.TensorData {
	values := func(count, modulus int, scale float32) []float32 {
		if !patterned {
			return nil
		}
		result := make([]float32, count)
		for index := range result {
			result[index] = float32(index%modulus-modulus/2) * scale
		}
		return result
	}
	ones2 := []float32{1, 1}
	return []gguf.TensorData{
		f32Tensor("v.sam.pos_embd.weight", []uint64{2, 4, 4}, values(32, 7, .01)),
		f32Tensor("v.sam.patch_embd.weight", []uint64{16, 16, 3, 2}, values(1536, 7, .002)),
		f32Tensor("v.sam.patch_embd.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.sam.blk.0.pre_ln.weight", []uint64{2}, ones2), f32Tensor("v.sam.blk.0.pre_ln.bias", []uint64{2}, nil),
		f32Tensor("v.sam.blk.0.post_ln.weight", []uint64{2}, ones2), f32Tensor("v.sam.blk.0.post_ln.bias", []uint64{2}, nil),
		f32Tensor("v.sam.blk.0.attn.pos_h.weight", []uint64{2, 3}, values(6, 5, .01)),
		f32Tensor("v.sam.blk.0.attn.pos_w.weight", []uint64{2, 3}, values(6, 7, .01)),
		f32Tensor("v.sam.blk.0.attn.qkv.weight", []uint64{2, 6}, values(12, 7, .01)),
		f32Tensor("v.sam.blk.0.attn.qkv.bias", []uint64{6}, values(6, 5, .01)),
		f32Tensor("v.sam.blk.0.attn.out.weight", []uint64{2, 2}, values(4, 5, .01)),
		f32Tensor("v.sam.blk.0.attn.out.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.sam.blk.0.mlp.lin1.weight", []uint64{2, 3}, values(6, 7, .01)),
		f32Tensor("v.sam.blk.0.mlp.lin1.bias", []uint64{3}, values(3, 5, .01)),
		f32Tensor("v.sam.blk.0.mlp.lin2.weight", []uint64{3, 2}, values(6, 7, .01)),
		f32Tensor("v.sam.blk.0.mlp.lin2.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.sam.neck.0.weight", []uint64{1, 1, 2, 2}, values(4, 5, .01)),
		f32Tensor("v.sam.neck.1.weight", []uint64{2}, ones2), f32Tensor("v.sam.neck.1.bias", []uint64{2}, nil),
		f32Tensor("v.sam.neck.2.weight", []uint64{3, 3, 2, 2}, values(36, 7, .01)),
		f32Tensor("v.sam.neck.3.weight", []uint64{2}, ones2), f32Tensor("v.sam.neck.3.bias", []uint64{2}, nil),
		f32Tensor("v.sam.net_2.weight", []uint64{3, 3, 2, 2}, values(36, 5, .01)),
		f32Tensor("v.sam.net_3.weight", []uint64{3, 3, 2, 2}, values(36, 7, .01)),
		f32Tensor("v.class_embd", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.position_embd.weight", []uint64{2, 2}, values(4, 5, .01)),
		f32Tensor("v.blk.0.ln1.weight", []uint64{2}, ones2), f32Tensor("v.blk.0.ln1.bias", []uint64{2}, nil),
		f32Tensor("v.blk.0.ln2.weight", []uint64{2}, ones2), f32Tensor("v.blk.0.ln2.bias", []uint64{2}, nil),
		f32Tensor("v.blk.0.attn_qkv.weight", []uint64{2, 6}, values(12, 7, .01)),
		f32Tensor("v.blk.0.attn_qkv.bias", []uint64{6}, values(6, 5, .01)),
		f32Tensor("v.blk.0.attn_out.weight", []uint64{2, 2}, values(4, 5, .01)),
		f32Tensor("v.blk.0.attn_out.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("v.blk.0.ffn_up.weight", []uint64{2, 3}, values(6, 7, .01)),
		f32Tensor("v.blk.0.ffn_up.bias", []uint64{3}, values(3, 5, .01)),
		f32Tensor("v.blk.0.ffn_down.weight", []uint64{3, 2}, values(6, 7, .01)),
		f32Tensor("v.blk.0.ffn_down.bias", []uint64{2}, values(2, 5, .01)),
		f32Tensor("mm.model.fc.weight", []uint64{4, 3}, values(12, 7, .01)),
		f32Tensor("mm.model.fc.bias", []uint64{3}, values(3, 5, .01)),
		f32Tensor("v.image_newline", []uint64{3}, nil), f32Tensor("v.view_seperator", []uint64{3}, nil),
	}
}
