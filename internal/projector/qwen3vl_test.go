package projector

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	cudatest "overgo/internal/cuda/testutil"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type qwen3VLPromptTokenizer struct{}

func (qwen3VLPromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		if strings.HasPrefix(text, Qwen3VLImagePad) {
			ids, text = append(ids, 8), text[len(Qwen3VLImagePad):]
		} else {
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestQwen3VLRealFixture(t *testing.T) {
	classifyProjectorIntegration(t)
	projectorPath := os.Getenv("OVERGO_QWEN35_MMPROJ")
	imagePath := os.Getenv("OVERGO_QWEN35_IMAGE")
	goldenPath := os.Getenv("OVERGO_QWEN35_GOLDEN")
	if projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set OVERGO_QWEN35_MMPROJ, OVERGO_QWEN35_IMAGE, and OVERGO_QWEN35_GOLDEN")
	}
	type probeRecord struct {
		Shape      []int     `json:"shape"`
		ProbeIndex []int     `json:"probe_index"`
		ProbeValue []float32 `json:"probe_value"`
		L2         float64   `json:"l2"`
	}
	var golden struct {
		ImageGridTHW   []int       `json:"image_grid_thw"`
		NumImageTokens int         `json:"num_image_tokens"`
		PixelValues    probeRecord `json:"pixel_values"`
		Intermediates  struct {
			Merger probeRecord `json:"merger"`
		} `json:"vit_intermediates"`
	}
	if err := jsonfile.Decode(goldenPath, &golden); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	input, err := png.Decode(file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	runner, err := openQwen3VLFixture(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	processed, err := PreprocessQwen3VLImage(input, runner.Spec(), DefaultQwen3VLPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.ImageGridTHW) != 3 || processed.GridT != golden.ImageGridTHW[0] ||
		processed.GridH != golden.ImageGridTHW[1] || processed.GridW != golden.ImageGridTHW[2] {
		t.Fatalf("grid = %d,%d,%d, want %v", processed.GridT, processed.GridH, processed.GridW, golden.ImageGridTHW)
	}
	if len(golden.PixelValues.Shape) != 2 || len(processed.PixelValues) != golden.PixelValues.Shape[0]*golden.PixelValues.Shape[1] {
		t.Fatalf("pixel values = %d, want shape %v", len(processed.PixelValues), golden.PixelValues.Shape)
	}
	compareProbes(t, "pixel values", processed.PixelValues, golden.PixelValues, 0.15)
	if os.Getenv("OVERGO_QWEN35_PROJECTOR_FULL") == "" {
		return
	}
	output, err := runner.EncodeImage(context.Background(), input, DefaultQwen3VLPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(uint64(runner.Spec().OutputHidden), uint64(golden.NumImageTokens))) {
		t.Fatalf("merger shape = %v, want [%d %d]", output.Embeddings.Shape, runner.Spec().OutputHidden, golden.NumImageTokens)
	}
	compareProbes(t, "merger", output.Embeddings.Data, golden.Intermediates.Merger, 0.35)
	var sumSquares float64
	for _, value := range output.Embeddings.Data {
		sumSquares += float64(value) * float64(value)
	}
	gotL2 := math.Sqrt(sumSquares)
	if relative := math.Abs(gotL2-golden.Intermediates.Merger.L2) / golden.Intermediates.Merger.L2; relative > 0.03 {
		t.Fatalf("merger L2 = %g, want %g (relative %g)", gotL2, golden.Intermediates.Merger.L2, relative)
	}
}

func compareProbes(t *testing.T, name string, values []float32, record struct {
	Shape      []int     `json:"shape"`
	ProbeIndex []int     `json:"probe_index"`
	ProbeValue []float32 `json:"probe_value"`
	L2         float64   `json:"l2"`
}, tolerance float32) {
	t.Helper()
	if len(record.ProbeIndex) != len(record.ProbeValue) {
		t.Fatalf("%s probe counts differ", name)
	}
	for index, offset := range record.ProbeIndex {
		if offset < 0 || offset >= len(values) {
			t.Fatalf("%s probe offset %d is outside %d values", name, offset, len(values))
		}
		if delta := float32(math.Abs(float64(values[offset] - record.ProbeValue[index]))); delta > tolerance {
			t.Fatalf("%s[%d] = %g, want %g (delta %g)", name, offset, values[offset], record.ProbeValue[index], delta)
		}
	}
}

func TestQwen3VLRunnerTinyFixture(t *testing.T) {
	metadata := tinyQwen3VLMetadata()
	tensors := tinyQwen3VLTensors()
	path := testutil.TempGGUF(t, "mmproj.gguf", metadata, tensors)
	runner, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{})
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
	output, err := runner.EncodeImage(context.Background(), input, Qwen3VLPreprocessOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(6, 1)) {
		t.Fatalf("output shape = %v", output.Embeddings.Shape)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for index, value := range output.Embeddings.Data {
		if value != want[index] {
			t.Fatalf("output[%d] = %g, want %g", index, value, want[index])
		}
	}
	if output.GridT != 1 || output.GridH != 2 || output.GridW != 2 || output.MergeSize != 2 {
		t.Fatalf("output grid = %d,%d,%d merge=%d", output.GridT, output.GridH, output.GridW, output.MergeSize)
	}
}

func TestQwen3VLDeepstackTinyFixture(t *testing.T) {
	metadata := slices.DeleteFunc(tinyQwen3VLDeepstackMetadata(), func(item gguf.Metadata) bool {
		return item.Key == "clip.vision.is_deepstack_layers"
	})
	path := testutil.TempGGUF(t, "mmproj.gguf", metadata, tinyQwen3VLDeepstackTensors())
	runner, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	if len(runner.Spec().DeepstackLayers) != 1 || !runner.Spec().DeepstackLayers[0] {
		t.Fatalf("deepstack flags = %v", runner.Spec().DeepstackLayers)
	}
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	output, err := runner.EncodeImage(context.Background(), input, Qwen3VLPreprocessOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(output.DeepstackEmbeddings) != 1 || !output.DeepstackEmbeddings[0].Shape.Equal(mustShape(6, 1)) {
		t.Fatalf("deepstack output = %+v", output.DeepstackEmbeddings)
	}
	want := []float32{7, 8, 9, 10, 11, 12}
	if !slices.Equal(output.DeepstackEmbeddings[0].Data, want) {
		t.Fatalf("deepstack output = %v, want %v", output.DeepstackEmbeddings[0].Data, want)
	}
}

func TestQwen3VLDeepstackCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := testutil.TempGGUF(t, "mmproj.gguf", tinyQwen3VLDeepstackMetadata(), tinyQwen3VLDeepstackTensors())
	cpu, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	options := Qwen3VLPreprocessOptions{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio}
	want, err := cpu.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cuda.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.DeepstackEmbeddings) != 1 || len(want.DeepstackEmbeddings) != 1 {
		t.Fatalf("deepstack streams = %d/%d", len(got.DeepstackEmbeddings), len(want.DeepstackEmbeddings))
	}
	compareFloat32Tolerance(t, "Qwen3-VL deepstack", got.DeepstackEmbeddings[0].Data, want.DeepstackEmbeddings[0].Data, 2e-3)
}

func TestQwen3VLMultipleImagePrompt(t *testing.T) {
	metadata := tinyQwen3VLDeepstackMetadata()
	for index := range metadata {
		switch metadata[index].Key {
		case "clip.vision.image_size":
			metadata[index].Value.Data = uint32(256)
		case "clip.vision.patch_size":
			metadata[index].Value.Data = uint32(128)
		}
	}
	tensors := tinyQwen3VLDeepstackTensors()
	tensors[0] = f32Tensor("v.patch_embd.weight", []uint64{128, 128, 3, 4}, nil)
	tensors[1] = f32Tensor("v.patch_embd.weight.1", []uint64{128, 128, 3, 4}, nil)
	path := testutil.TempGGUF(t, "mmproj.gguf", metadata, tensors)
	runner, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	prompt, err := runner.BuildImagesPrompt(
		context.Background(), qwen3VLPromptTokenizer{}, []image.Image{input, input},
		[]string{"A", "B", "C"}, PromptOptions{Thinking: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.EmbeddingTokenIndices) != 2 || len(prompt.Embeddings) != 12 || prompt.EmbeddingWidth != 6 {
		t.Fatalf("multi-image projection = indices %v embeddings %d width %d", prompt.EmbeddingTokenIndices, len(prompt.Embeddings), prompt.EmbeddingWidth)
	}
	if len(prompt.DeepstackEmbeddings) != 1 || len(prompt.DeepstackEmbeddings[0]) != 12 {
		t.Fatalf("multi-image deepstack streams = %v", prompt.DeepstackEmbeddings)
	}
	for axis := range prompt.MultiAxisPositions {
		if len(prompt.MultiAxisPositions[axis]) != len(prompt.TokenIDs) {
			t.Fatalf("axis %d positions = %d, tokens %d", axis, len(prompt.MultiAxisPositions[axis]), len(prompt.TokenIDs))
		}
	}
}

func TestQwen3VLRunnerTinyFixtureCUDAMatchesCPU(t *testing.T) {
	cudatest.Require(t)
	path := testutil.TempGGUF(t, "mmproj.gguf", tinyQwen3VLMetadata(), nonzeroTinyQwen3VLTensors())
	cpu, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	cuda, err := openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{CUDA: true})
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
	options := Qwen3VLPreprocessOptions{MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio}
	wantImage, err := cpu.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	gotImage, err := cuda.EncodeImage(context.Background(), input, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Qwen3-VL image", gotImage.Embeddings.Data, wantImage.Embeddings.Data, 2e-3)
	frames := []image.Image{input, input, input, input}
	wantVideo, err := cpu.EncodeFrames(context.Background(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	gotVideo, err := cuda.EncodeFrames(context.Background(), frames, options)
	if err != nil {
		t.Fatal(err)
	}
	compareFloat32Tolerance(t, "Qwen3-VL video", gotVideo.Embeddings.Data, wantVideo.Embeddings.Data, 2e-3)
}

func compareFloat32Tolerance(t *testing.T, name string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s values = %d, want %d", name, len(got), len(want))
	}
	for index := range want {
		if delta := float32(math.Abs(float64(got[index] - want[index]))); delta > tolerance {
			t.Fatalf("%s[%d] = %g, want %g (delta %g)", name, index, got[index], want[index], delta)
		}
	}
}

func TestPreprocessQwen3VLImageMergedOrder(t *testing.T) {
	spec := Qwen3VLSpec{
		visionBackboneSpec: fixtureVisionBackbone(4, 2, 4, 8, 1, 1),
		MergerIntermediate: 16, OutputHidden: 6, MergeSize: 2,
	}
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(y*4 + x), A: fixtureOpaqueAlpha})
		}
	}
	processed, err := PreprocessQwen3VLImage(input, spec, Qwen3VLPreprocessOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(processed.PixelValues) != 4*2*3*2*2 {
		t.Fatalf("pixel values = %d", len(processed.PixelValues))
	}
	patchWidth := 24
	if processed.PixelValues[0] != 0 || processed.PixelValues[patchWidth] != float32(2)/255 ||
		processed.PixelValues[2*patchWidth] != float32(8)/255 || processed.PixelValues[3*patchWidth] != float32(10)/255 {
		t.Fatalf("merged patch starts = %g %g %g %g",
			processed.PixelValues[0], processed.PixelValues[patchWidth],
			processed.PixelValues[2*patchWidth], processed.PixelValues[3*patchWidth])
	}
}

func TestPreprocessQwen3VLFramesTemporalOrder(t *testing.T) {
	spec := Qwen3VLSpec{
		visionBackboneSpec: fixtureVisionBackbone(4, 2, 4, 8, 1, 1),
		MergerIntermediate: 16, OutputHidden: 6, MergeSize: 2,
	}
	frames := make([]image.Image, 3)
	for index, red := range []uint8{10, 20, 30} {
		frame := image.NewRGBA(image.Rect(0, 0, 4, 4))
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				frame.SetRGBA(x, y, color.RGBA{R: red, A: fixtureOpaqueAlpha})
			}
		}
		frames[index] = frame
	}
	processed, err := PreprocessQwen3VLFrames(frames, spec, Qwen3VLPreprocessOptions{
		MinPixels: fixtureSmallPixelBudget, MaxPixels: fixtureSmallPixelBudget, MaxAspectRatio: fixtureMaxAspectRatio,
	})
	if err != nil {
		t.Fatal(err)
	}
	if processed.GridT != 2 || processed.GridH != 2 || processed.GridW != 2 {
		t.Fatalf("grid = %d,%d,%d", processed.GridT, processed.GridH, processed.GridW)
	}
	const patchWidth = 24
	if processed.PixelValues[0] != float32(10)/255 || processed.PixelValues[4] != float32(20)/255 ||
		processed.PixelValues[4*patchWidth] != float32(30)/255 || processed.PixelValues[4*patchWidth+4] != float32(30)/255 {
		t.Fatalf("temporal values = %g %g %g %g", processed.PixelValues[0], processed.PixelValues[4],
			processed.PixelValues[4*patchWidth], processed.PixelValues[4*patchWidth+4])
	}
}

func TestQwen3VLRealVideoFixture(t *testing.T) {
	classifyProjectorIntegration(t)
	projectorPath := os.Getenv("OVERGO_QWEN35_MMPROJ")
	goldenPath := os.Getenv("OVERGO_QWEN35_VIDEO_GOLDEN")
	if projectorPath == "" || goldenPath == "" {
		t.Skip("set OVERGO_QWEN35_MMPROJ and OVERGO_QWEN35_VIDEO_GOLDEN")
	}
	type probeRecord struct {
		Shape      []int     `json:"shape"`
		ProbeIndex []int     `json:"probe_index"`
		ProbeValue []float32 `json:"probe_value"`
		L2         float64   `json:"l2"`
	}
	var golden struct {
		VideoGridTHW   []int       `json:"video_grid_thw"`
		NumVideoTokens int         `json:"num_video_tokens"`
		PromptText     string      `json:"prompt_text"`
		Question       string      `json:"question"`
		PixelValues    probeRecord `json:"pixel_values_videos"`
		Intermediates  struct {
			Merger probeRecord `json:"merger"`
		} `json:"vit_intermediates"`
	}
	if err := jsonfile.Decode(goldenPath, &golden); err != nil {
		t.Fatal(err)
	}
	frames := make([]image.Image, 16)
	for temporal := range frames {
		frame := image.NewRGBA(image.Rect(0, 0, 224, 224))
		for y := 0; y < 224; y++ {
			for x := 0; x < 224; x++ {
				frame.SetRGBA(x, y, color.RGBA{
					R: uint8((x*4 + temporal*8) % fixtureChannelModulus),
					G: uint8((y*5 + temporal*4) % fixtureChannelModulus),
					B: uint8(((x+y)*3 + temporal*16) % fixtureChannelModulus), A: fixtureOpaqueAlpha,
				})
			}
		}
		frames[temporal] = frame
	}
	runner, err := openQwen3VLFixture(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	processed, err := PreprocessQwen3VLFrames(frames, runner.Spec(), DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.VideoGridTHW) != 3 || processed.GridT != golden.VideoGridTHW[0] ||
		processed.GridH != golden.VideoGridTHW[1] || processed.GridW != golden.VideoGridTHW[2] {
		t.Fatalf("grid = %d,%d,%d, want %v", processed.GridT, processed.GridH, processed.GridW, golden.VideoGridTHW)
	}
	if prompt := Qwen35VideoPromptText("", golden.Question, processed.GridT,
		(processed.GridH/runner.Spec().MergeSize)*(processed.GridW/runner.Spec().MergeSize), 24, true); prompt != golden.PromptText {
		t.Fatal("video prompt differs from golden")
	}
	compareProbes(t, "video pixel values", processed.PixelValues, golden.PixelValues, 0.15)
	if os.Getenv("OVERGO_QWEN35_VIDEO_FULL") == "" {
		return
	}
	output, err := runner.EncodeFrames(context.Background(), frames, DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(uint64(runner.Spec().OutputHidden), uint64(golden.NumVideoTokens))) {
		t.Fatalf("merger shape = %v", output.Embeddings.Shape)
	}
	compareProbes(t, "video merger", output.Embeddings.Data, golden.Intermediates.Merger, 0.35)
	var sumSquares float64
	for _, value := range output.Embeddings.Data {
		sumSquares += float64(value) * float64(value)
	}
	if relative := math.Abs(math.Sqrt(sumSquares)-golden.Intermediates.Merger.L2) / golden.Intermediates.Merger.L2; relative > 0.03 {
		t.Fatalf("video merger L2 relative = %g", relative)
	}
}

func tinyQwen3VLMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: qwen3VLProjectorType}},
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
		{Key: "clip.vision.is_deepstack_layers", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: []bool{false}}},
	}
}

func openQwen3VLFixture(path string) (*Qwen3VLRunner, error) {
	return openImageProjectorAs[*Qwen3VLRunner](path, OpenOptions{
		CUDA: os.Getenv("OVERGO_QWEN35_PROJECTOR_CUDA") != "",
	})

}

func tinyQwen3VLDeepstackMetadata() []gguf.Metadata {
	metadata := tinyQwen3VLMetadata()
	for index := range metadata {
		if metadata[index].Key == "clip.vision.is_deepstack_layers" {
			metadata[index].Value.Data = []bool{true}
		}
	}
	return metadata
}

func tinyQwen3VLTensors() []gguf.TensorData {
	tensors := []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.weight.1", []uint64{2, 2, 3, 4}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{4}, nil),
		f32Tensor("v.position_embd.weight", []uint64{4, 4}, nil),
		f32Tensor("v.post_ln.weight", []uint64{4}, []float32{1, 1, 1, 1}),
		f32Tensor("v.post_ln.bias", []uint64{4}, nil),
		f32Tensor("mm.0.weight", []uint64{16, 16}, nil),
		f32Tensor("mm.0.bias", []uint64{16}, nil),
		f32Tensor("mm.2.weight", []uint64{16, 6}, nil),
		f32Tensor("mm.2.bias", []uint64{6}, []float32{1, 2, 3, 4, 5, 6}),
	}
	for name, shape := range map[string][]uint64{
		"attn_qkv.weight": {4, 12}, "attn_qkv.bias": {12},
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

func tinyQwen3VLDeepstackTensors() []gguf.TensorData {
	tensors := tinyQwen3VLTensors()
	tensors = append(tensors,
		f32Tensor("v.deepstack.0.norm.weight", []uint64{16}, slices.Repeat([]float32{1}, 16)),
		f32Tensor("v.deepstack.0.norm.bias", []uint64{16}, nil),
		f32Tensor("v.deepstack.0.fc1.weight", []uint64{16, 16}, nil),
		f32Tensor("v.deepstack.0.fc1.bias", []uint64{16}, nil),
		f32Tensor("v.deepstack.0.fc2.weight", []uint64{16, 6}, nil),
		f32Tensor("v.deepstack.0.fc2.bias", []uint64{6}, []float32{7, 8, 9, 10, 11, 12}),
	)
	return tensors
}

func nonzeroTinyQwen3VLTensors() []gguf.TensorData {
	base := tinyQwen3VLTensors()
	result := make([]gguf.TensorData, len(base))
	for tensorIndex, item := range base {
		elements := uint64(1)
		for _, dimension := range item.Shape {
			elements *= dimension
		}
		values := make([]float32, int(elements))
		isNorm := strings.HasSuffix(item.Name, "ln.weight") || strings.Contains(item.Name, "ln1.weight") ||
			strings.Contains(item.Name, "ln2.weight")
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

func f32Tensor(name string, shape []uint64, values []float32) gguf.TensorData {
	elements := uint64(1)
	for _, dimension := range shape {
		elements *= dimension
	}
	if values == nil {
		values = make([]float32, int(elements))
	}
	var data bytes.Buffer
	_ = binary.Write(&data, binary.LittleEndian, values)
	return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(data.Bytes())}
}

func mustShape(dimensions ...uint64) tensor.Shape {
	return tensor.MustShape(dimensions...)
}
