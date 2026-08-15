package projector

import (
	"context"
	"image"
	"image/color"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type qwen2VLPromptTokenizer struct {
	text string
}

func (t *qwen2VLPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	t.text = text
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		switch {
		case len(text) >= len(Qwen3VLImagePad) && text[:len(Qwen3VLImagePad)] == Qwen3VLImagePad:
			ids, text = append(ids, 8), text[len(Qwen3VLImagePad):]
		case len(text) >= len(Qwen3VLVideoPad) && text[:len(Qwen3VLVideoPad)] == Qwen3VLVideoPad:
			ids, text = append(ids, 9), text[len(Qwen3VLVideoPad):]
		default:
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestQwen2VLRunnerTinyFixture(t *testing.T) {
	path := testutil.TempGGUF(t, "mmproj.gguf", tinyQwen2VLMetadata(), tinyQwen2VLTensors())
	runner, err := openImageProjectorAs[*Qwen2VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 40), G: uint8(y * 40), B: 80, A: fixtureOpaqueAlpha})
		}
	}
	output, err := runner.EncodeImage(context.Background(), input, Qwen2VLPreprocessOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(6, 1)) {
		t.Fatalf("output shape = %v", output.Embeddings.Shape)
	}
	for index, want := range []float32{1, 2, 3, 4, 5, 6} {
		if output.Embeddings.Data[index] != want {
			t.Fatalf("output[%d] = %g, want %g", index, output.Embeddings.Data[index], want)
		}
	}
	if output.GridT != 1 || output.GridH != 2 || output.GridW != 2 || output.MergeSize != 2 {
		t.Fatalf("output grid = %d,%d,%d merge=%d", output.GridT, output.GridH, output.GridW, output.MergeSize)
	}
}

func TestQwen2VLSpecOptionalNormPairs(t *testing.T) {
	tensors := tinyQwen2VLTensors()
	tensors = append(tensors, f32Tensor("v.pre_ln.weight", []uint64{4}, []float32{1, 1, 1, 1}))
	path := testutil.TempGGUF(t, "mmproj.gguf", tinyQwen2VLMetadata(), tensors)
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if _, err := ReadQwen2VLSpec(opened); err == nil {
		t.Fatal("expected incomplete optional pre-LayerNorm rejection")
	}
}

func TestQwen2VLLegacyFFNNamesAndDefaultMerge(t *testing.T) {
	metadata := tinyQwen2VLMetadata()
	for index := range metadata {
		if metadata[index].Key == "clip.vision.spatial_merge_size" {
			metadata = append(metadata[:index], metadata[index+1:]...)
			break
		}
	}
	tensors := tinyQwen2VLTensors()
	for index := range tensors {
		switch tensors[index].Name {
		case "v.blk.0.ffn_up.weight":
			tensors[index] = f32Tensor(tensors[index].Name, []uint64{8, 4}, nil)
		case "v.blk.0.ffn_up.bias":
			tensors[index] = f32Tensor(tensors[index].Name, []uint64{4}, nil)
		case "v.blk.0.ffn_down.weight":
			tensors[index] = f32Tensor(tensors[index].Name, []uint64{4, 8}, nil)
		case "v.blk.0.ffn_down.bias":
			tensors[index] = f32Tensor(tensors[index].Name, []uint64{8}, nil)
		}
	}
	path := testutil.TempGGUF(t, "mmproj.gguf", metadata, tensors)
	runner, err := openImageProjectorAs[*Qwen2VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	if !runner.Spec().LegacyFFNSwapped || runner.Spec().MergeSize != 2 {
		t.Fatalf("legacy/default spec = %+v", runner.Spec())
	}
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	if _, err := runner.EncodeImage(context.Background(), input, Qwen2VLPreprocessOptions{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio}); err != nil {
		t.Fatal(err)
	}
}

func TestQwen2VLImageAndVideoPrompts(t *testing.T) {
	metadata := tinyQwen2VLMetadata()
	for index := range metadata {
		switch metadata[index].Key {
		case "clip.vision.image_size":
			metadata[index].Value.Data = uint32(256)
		case "clip.vision.patch_size":
			metadata[index].Value.Data = uint32(128)
		}
	}
	tensors := tinyQwen2VLTensors()
	tensors[0] = f32Tensor("v.patch_embd.weight", []uint64{128, 128, 3, 4}, nil)
	tensors[1] = f32Tensor("v.patch_embd.weight.1", []uint64{128, 128, 3, 4}, nil)
	path := testutil.TempGGUF(t, "mmproj.gguf", metadata, tensors)
	opened, err := OpenAs[ImageProjector](context.Background(), path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	runner, ok := opened.(*Qwen2VLRunner)
	if !ok {
		t.Fatalf("projector type = %T", opened)
	}
	defer runner.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	tok := &qwen2VLPromptTokenizer{}
	prompt, err := runner.BuildImagePrompt(context.Background(), tok, input, "before", "after", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.EmbeddingTokenIndices) != 1 || prompt.EmbeddingWidth != 6 || len(prompt.Embeddings) != 6 {
		t.Fatalf("image projection = indices %v width %d values %d", prompt.EmbeddingTokenIndices, prompt.EmbeddingWidth, len(prompt.Embeddings))
	}
	if len(tok.text) == 0 {
		t.Fatal("image prompt text is empty")
	}
	frames := []image.Image{input, input, input, input}
	video, err := runner.BuildVideoPrompt(context.Background(), tok, frames, "", "describe", 24, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(video.EmbeddingTokenIndices) != 2 || len(video.Embeddings) != 12 {
		t.Fatalf("video projection = indices %v values %d", video.EmbeddingTokenIndices, len(video.Embeddings))
	}
	first := int(video.EmbeddingTokenIndices[0])
	second := int(video.EmbeddingTokenIndices[1])
	if video.MultiAxisPositions[0][second] != video.MultiAxisPositions[0][first]+1 {
		t.Fatalf("video temporal positions = %d,%d", video.MultiAxisPositions[0][first], video.MultiAxisPositions[0][second])
	}
}

func TestQwen2VLVideoPositions(t *testing.T) {
	positions, err := Qwen2VLVideoPositions(12, 2, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if positions[0][2] != 2 || positions[0][6] != 3 || positions[1][4] != 3 ||
		positions[2][3] != 3 || positions[3][9] != 0 || positions[0][10] != 4 {
		t.Fatalf("unexpected Qwen2-VL video positions: %v", positions)
	}
}

func TestQwen2VLRunnerTinyFixtureCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := testutil.TempGGUF(t, "mmproj.gguf", tinyQwen2VLMetadata(), nonzeroTinyQwen2VLTensors())
	cpu, err := openImageProjectorAs[*Qwen2VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*Qwen2VLRunner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 40), G: uint8(y * 40), B: 80, A: fixtureOpaqueAlpha})
		}
	}
	options := Qwen2VLPreprocessOptions{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio}
	wantImage, err := cpu.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	gotImage, err := cuda.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Qwen2-VL image", gotImage.Embeddings.Data, wantImage.Embeddings.Data, 2e-3)
	frames := []image.Image{input, input, input, input}
	wantVideo, err := cpu.EncodeFrames(context.Background(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	gotVideo, err := cuda.EncodeFrames(context.Background(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Qwen2-VL video", gotVideo.Embeddings.Data, wantVideo.Embeddings.Data, 2e-3)
}

func tinyQwen2VLMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: qwen2VLProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.use_gelu", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.image_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(4)}},
		{Key: "clip.vision.feed_forward_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(8)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(6)}},
		{Key: "clip.vision.block_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.spatial_merge_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.vision.image_mean", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{0, 0, 0}}},
		{Key: "clip.vision.image_std", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{1, 1, 1}}},
	}
}

func tinyQwen2VLTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.weight.1", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("mm.0.weight", []uint64{16, 16}, nil),
		f32Tensor("mm.0.bias", []uint64{16}, nil),
		f32Tensor("mm.2.weight", []uint64{16, 6}, nil),
		f32Tensor("mm.2.bias", []uint64{6}, []float32{1, 2, 3, 4, 5, 6}),
	}
	for name, shape := range map[string][]uint64{
		"attn_q.weight": {4, 4}, "attn_q.bias": {4},
		"attn_k.weight": {4, 4}, "attn_k.bias": {4},
		"attn_v.weight": {4, 4}, "attn_v.bias": {4},
		"attn_out.weight": {4, 4}, "attn_out.bias": {4},
		"ffn_up.weight": {4, 8}, "ffn_up.bias": {8},
		"ffn_down.weight": {8, 4}, "ffn_down.bias": {4},
		"ln1.weight": {4}, "ln1.bias": {4}, "ln2.weight": {4}, "ln2.bias": {4},
	} {
		values := []float32(nil)
		if name == "ln1.weight" || name == "ln2.weight" {
			values = []float32{1, 1, 1, 1}
		}
		tensors = append(tensors, f32Tensor("v.blk.0."+name, shape, values))
	}
	return tensors
}

func nonzeroTinyQwen2VLTensors() []gguf.TensorData {
	base := tinyQwen2VLTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := item.Name == "v.pre_ln.weight" || item.Name == "v.post_ln.weight" ||
			len(item.Name) >= len("ln1.weight") && item.Name[len(item.Name)-len("ln1.weight"):] == "ln1.weight" ||
			len(item.Name) >= len("ln2.weight") && item.Name[len(item.Name)-len("ln2.weight"):] == "ln2.weight"
		for index := range values {
			value := float32((index+tensorIndex*3)%13-6) * 0.0075
			if isNorm {
				value += 1
			}
			values[index] = value
		}
		result[tensorIndex] = f32Tensor(item.Name, item.Shape, values)
	}
	return result
}
