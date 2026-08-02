package projector

import (
	"context"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "llamacpp2go/internal/cuda/testutil"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

type granite4VisionPromptTokenizer struct{}

func (granite4VisionPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	const placeholder = "\x00"
	text = strings.ReplaceAll(text, Granite4VisionImageToken, placeholder)
	ids := make([]tokenizer.TokenID, 0, len(text))
	for _, value := range text {
		if value == 0 {
			ids = append(ids, 9352)
		} else {
			ids = append(ids, tokenizer.TokenID(value))
		}
	}
	return ids, nil
}

func TestGranite4VisionRunnerTinyFixture(t *testing.T) {
	runner, err := OpenGranite4Vision(writeTinyGranite4Vision(t, tinyGranite4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	output, err := runner.EncodeImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 4, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(4, 2)) || len(output.DeepstackEmbeddings) != 1 ||
		!output.DeepstackEmbeddings[0].Shape.Equal(mustShape(4, 2)) {
		t.Fatalf("output = base %v deepstack %+v", output.Embeddings.Shape, output.DeepstackEmbeddings)
	}
	if !slices.Equal(output.Embeddings.Data, make([]float32, 8)) ||
		!slices.Equal(output.DeepstackEmbeddings[0].Data, make([]float32, 8)) {
		t.Fatalf("output values are nonzero")
	}
	if output.TileCount != 1 || output.GridH != 0 || output.GridW != 0 {
		t.Fatalf("tiles=%d grid=%d,%d", output.TileCount, output.GridH, output.GridW)
	}
}

func TestOpenImageProjectorDispatchesGranite4Vision(t *testing.T) {
	projector, err := OpenImageProjector(writeTinyGranite4Vision(t, tinyGranite4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer projector.Close()
	if _, ok := projector.(*Granite4VisionRunner); !ok {
		t.Fatalf("projector type = %T", projector)
	}
}

func TestPreprocessGranite4VisionOverviewAndTileNewlines(t *testing.T) {
	input := image.NewRGBA(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 30), A: 255})
		}
	}
	processed, err := PreprocessGranite4VisionImage(input, tinyGranite4VisionSpec())
	if err != nil {
		t.Fatal(err)
	}
	if processed.GridW != 2 || processed.GridH != 1 || len(processed.Tiles) != 3 {
		t.Fatalf("grid=%d,%d tiles=%d", processed.GridW, processed.GridH, len(processed.Tiles))
	}
	if processed.Tiles[0].AddNewline || !processed.Tiles[1].AddNewline || !processed.Tiles[2].AddNewline {
		t.Fatalf("newline flags = %v,%v,%v", processed.Tiles[0].AddNewline, processed.Tiles[1].AddNewline, processed.Tiles[2].AddNewline)
	}
	if processed.Tiles[1].PixelValues[0] >= processed.Tiles[2].PixelValues[0] {
		t.Fatalf("grid tile order is not left-to-right")
	}
}

func TestGranite4VisionPromptCarriesDeepstack(t *testing.T) {
	runner, err := OpenGranite4Vision(writeTinyGranite4Vision(t, tinyGranite4VisionTensors()))
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	prompt, err := runner.BuildImagePrompt(
		context.Background(), granite4VisionPromptTokenizer{}, image.NewRGBA(image.Rect(0, 0, 4, 4)), "", "describe", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 4 || len(prompt.Embeddings) != 8 || len(prompt.DeepstackEmbeddings) != 1 ||
		len(prompt.DeepstackEmbeddings[0]) != 8 || len(prompt.EmbeddingTokenIndices) != 2 {
		t.Fatalf("prompt contract = width %d base %d deepstack %v indices %v", prompt.EmbeddingWidth, len(prompt.Embeddings), prompt.DeepstackEmbeddings, prompt.EmbeddingTokenIndices)
	}
	if prompt.EmbeddingTokenIndices[0] != uint32(prompt.EmbeddingStart) {
		t.Fatalf("start=%d indices=%v", prompt.EmbeddingStart, prompt.EmbeddingTokenIndices)
	}
}

func TestGranite4VisionCatalogRejectsMissingQFormerTensor(t *testing.T) {
	tensors := slices.DeleteFunc(tinyGranite4VisionTensors(), func(item gguf.TensorData) bool {
		return item.Name == "v.proj_blk.1.cross_attn_q.weight"
	})
	if _, err := OpenGranite4Vision(writeTinyGranite4Vision(t, tensors)); err == nil || !strings.Contains(err.Error(), "missing tensor") {
		t.Fatalf("error = %v", err)
	}
}

func TestGranite4VisionCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := writeTinyGranite4Vision(t, nonzeroGranite4VisionTensors(true))
	rewriteGranite4VisionMetadata(t, path, granite4VisionMultiwindowMetadata(), nonzeroGranite4VisionTensors(true))
	cpu, err := OpenGranite4Vision(path)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := OpenGranite4VisionWithOptions(path, Granite4VisionOpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 29), G: uint8(y * 31), B: uint8((x + y) * 13), A: 255})
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
	compareFloat32Tolerance(t, "Granite 4 Vision", got.Embeddings.Data, want.Embeddings.Data, 3e-3)
	if len(got.DeepstackEmbeddings) != len(want.DeepstackEmbeddings) {
		t.Fatalf("deepstack streams = %d, want %d", len(got.DeepstackEmbeddings), len(want.DeepstackEmbeddings))
	}
	for index := range got.DeepstackEmbeddings {
		compareFloat32Tolerance(t, "Granite 4 Vision deepstack", got.DeepstackEmbeddings[index].Data, want.DeepstackEmbeddings[index].Data, 3e-3)
	}
}

func writeTinyGranite4Vision(t *testing.T, tensors []gguf.TensorData) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "granite4-vision-mmproj.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, tinyGranite4VisionMetadata(), tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func rewriteGranite4VisionMetadata(t *testing.T, path string, metadata []gguf.Metadata, tensors []gguf.TensorData) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func tinyGranite4VisionMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: granite4VisionProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(64)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.projector.window_side", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.projector.query_side", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
		{Key: "clip.vision.feature_layer", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{0, 0}}},
		{Key: "clip.vision.projector.spatial_offsets", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{-1, 0}}},
		{Key: "clip.vision.image_grid_pinpoints", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{4, 4, 8, 4}}},
	}
}

func granite4VisionMultiwindowMetadata() []gguf.Metadata {
	metadata := tinyGranite4VisionMetadata()
	for index := range metadata {
		switch metadata[index].Key {
		case "clip.vision.image_size":
			metadata[index].Value.Data = uint32(8)
		case "clip.vision.image_grid_pinpoints":
			metadata[index].Value.Data = []int32{8, 8}
		}
	}
	return metadata
}

func tinyGranite4VisionSpec() Granite4VisionSpec {
	return Granite4VisionSpec{
		ImageSize: 4, PatchSize: 2, Hidden: 64, Intermediate: 8, ProjectionDim: 4, QFormerWidth: 8,
		Layers: 1, Heads: 1, WindowSide: 2, QuerySide: 1, LayerNormEpsilon: 1e-6,
		ImageStd: [3]float32{1, 1, 1}, FeatureLayers: []int{0, 0}, SpatialOffsets: []int{-1, 0},
		GridCandidates: []Granite4VisionResolution{{Width: 4, Height: 4}, {Width: 8, Height: 4}},
	}
}

func tinyGranite4VisionTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 64}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{64}, nil),
		f32Tensor("v.position_embd.weight", []uint64{64, 4}, nil),
		f32Tensor("v.image_newline", []uint64{4}, nil),
	}
	for name, shape := range map[string][]uint64{
		"attn_q.weight": {64, 64}, "attn_q.bias": {64},
		"attn_k.weight": {64, 64}, "attn_k.bias": {64},
		"attn_v.weight": {64, 64}, "attn_v.bias": {64},
		"attn_out.weight": {64, 64}, "attn_out.bias": {64},
		"ffn_up.weight": {64, 8}, "ffn_up.bias": {8},
		"ffn_down.weight": {8, 64}, "ffn_down.bias": {64},
		"ln1.weight": {64}, "ln1.bias": {64}, "ln2.weight": {64}, "ln2.bias": {64},
	} {
		values := []float32(nil)
		if name == "ln1.weight" || name == "ln2.weight" {
			values = slices.Repeat([]float32{1}, 64)
		}
		tensors = append(tensors, f32Tensor("v.blk.0."+name, shape, values))
	}
	for block := 0; block < 2; block++ {
		prefix := "v.proj_blk." + string(rune('0'+block)) + "."
		tensors = append(tensors,
			f32Tensor(prefix+"img_pos", []uint64{64, 4}, nil),
			f32Tensor(prefix+"query", []uint64{64, 1}, nil),
			f32Tensor(prefix+"linear.weight", []uint64{64, 4}, nil),
			f32Tensor(prefix+"linear.bias", []uint64{4}, nil),
		)
		for _, name := range []string{"norm", "post_norm", "self_attn_norm", "cross_attn_norm", "ffn_norm"} {
			tensors = append(tensors,
				f32Tensor(prefix+name+".weight", []uint64{64}, slices.Repeat([]float32{1}, 64)),
				f32Tensor(prefix+name+".bias", []uint64{64}, nil),
			)
		}
		for _, name := range []string{
			"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
			"cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out",
		} {
			tensors = append(tensors,
				f32Tensor(prefix+name+".weight", []uint64{64, 64}, nil),
				f32Tensor(prefix+name+".bias", []uint64{64}, nil),
			)
		}
		tensors = append(tensors,
			f32Tensor(prefix+"ffn_up.weight", []uint64{64, 8}, nil),
			f32Tensor(prefix+"ffn_up.bias", []uint64{8}, nil),
			f32Tensor(prefix+"ffn_down.weight", []uint64{8, 64}, nil),
			f32Tensor(prefix+"ffn_down.bias", []uint64{64}, nil),
		)
	}
	return tensors
}

func nonzeroGranite4VisionTensors(multiwindow bool) []gguf.TensorData {
	base := tinyGranite4VisionTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		shape := item.Shape
		if multiwindow && item.Name == "v.position_embd.weight" {
			shape = []uint64{64, 16}
		}
		elements := uint64(1)
		for _, dimension := range shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNormWeight := strings.HasSuffix(item.Name, "norm.weight") || strings.HasSuffix(item.Name, "ln1.weight") || strings.HasSuffix(item.Name, "ln2.weight")
		for index := range values {
			value := float32((index+tensorIndex*5)%17-8) * 0.004
			if isNormWeight {
				value += 1
			}
			values[index] = value
		}
		result[tensorIndex] = f32Tensor(item.Name, shape, values)
	}
	return result
}
