package projector

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

type gemma4PromptTokenizer struct{}

func (gemma4PromptTokenizer) TokenizeText(text string, _, _ bool) ([]tokenizer.TokenID, error) {
	if text == "<|image|>" {
		return []tokenizer.TokenID{8}, nil
	}
	if text == "<|video|>" {
		return []tokenizer.TokenID{9}, nil
	}
	ids := make([]tokenizer.TokenID, 0, len(text))
	for len(text) > 0 {
		switch {
		case strings.HasPrefix(text, "<|image|>"):
			ids, text = append(ids, 8), text[len("<|image|>"):]
		case strings.HasPrefix(text, "<|video|>"):
			ids, text = append(ids, 9), text[len("<|video|>"):]
		default:
			ids, text = append(ids, 1), text[1:]
		}
	}
	return ids, nil
}

func TestGemma4RunnerTinyFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, tinyGemma4Metadata(), tinyGemma4Tensors(), gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	runner, err := OpenGemma4(path)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	input := image.NewRGBA(image.Rect(0, 0, 3, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(x * 50), G: uint8(y * 50), B: 70, A: 255})
		}
	}
	output, err := runner.EncodeImage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	rows := output.GridH * output.GridW
	if rows <= 0 || rows > 280 || !output.Embeddings.Shape.Equal(mustShape(2, uint64(rows))) {
		t.Fatalf("output = grid %dx%d shape %v", output.GridH, output.GridW, output.Embeddings.Shape)
	}
	for index, value := range output.Embeddings.Data {
		if value != 0 {
			t.Fatalf("output[%d] = %g", index, value)
		}
	}
}

func TestGemma4VideoPromptBuildsFrameBlocks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = gguf.Write(file, tinyGemma4Metadata(), tinyGemma4Tensors(), gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err = file.Close(); err != nil {
		t.Fatal(err)
	}
	runner, err := OpenGemma4(path)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	frame := image.NewRGBA(image.Rect(0, 0, 3, 3))
	prompt, err := runner.BuildVideoPrompt(
		context.Background(), gemma4PromptTokenizer{}, []image.Image{frame, frame}, "", "Describe.", 2, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.AttentionBlocks) != 2 {
		t.Fatalf("attention blocks = %v", prompt.AttentionBlocks)
	}
	tokensPerFrame := int(prompt.AttentionBlocks[0].End - prompt.AttentionBlocks[0].Start)
	if tokensPerFrame <= 0 || tokensPerFrame > 70 ||
		prompt.AttentionBlocks[1].End-prompt.AttentionBlocks[1].Start != uint32(tokensPerFrame) {
		t.Fatalf("attention blocks = %v", prompt.AttentionBlocks)
	}
	if len(prompt.EmbeddingTokenIndices) != 2*tokensPerFrame ||
		len(prompt.Embeddings) != 2*tokensPerFrame*prompt.EmbeddingWidth {
		t.Fatalf("video projection size = indices %d embeddings %d width %d", len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings), prompt.EmbeddingWidth)
	}
	text := Gemma4VideoPromptText("Describe.", 2, tokensPerFrame, 2)
	if !strings.Contains(text, "00:00 <|image>") || strings.Count(text, "<|video|>") != 2*tokensPerFrame {
		t.Fatalf("video prompt = %q", text)
	}
}

func TestGemma4AudioTinyFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, tinyGemma4Metadata(), tinyGemma4Tensors(), gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	runner, err := OpenGemma4(path)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	spec, err := runner.AudioSpec()
	if err != nil {
		t.Fatal(err)
	}
	if spec.SampleRate != 16000 || spec.SamplesPerToken != 3 || spec.Hidden != 2 {
		t.Fatalf("audio spec = %+v", spec)
	}
	frames, rows, err := PreprocessGemma4Audio([]float32{1, 2, 3, 4}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if rows != 2 || len(frames) != 6 || frames[3] != 4 || frames[4] != 0 || frames[5] != 0 {
		t.Fatalf("audio frames = %v rows=%d", frames, rows)
	}
	output, err := runner.EncodeAudio(context.Background(), []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(2, 2)) {
		t.Fatalf("audio output shape = %v", output.Embeddings.Shape)
	}
	for index, value := range output.Embeddings.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("audio output[%d] = %g", index, value)
		}
	}
}

func TestPreprocessGemma4ImagePatchOrder(t *testing.T) {
	spec := Gemma4Spec{
		TeacherPatch: 1, PoolKernel: 2, ModelPatch: 2, PatchWidth: 12,
		Hidden: 2, PositionCount: 8, MaxImageTokens: 4,
		LayerNormEpsilon: 1e-5, RMSNormEpsilon: 1e-6,
	}
	input := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			input.SetRGBA(x, y, color.RGBA{R: uint8(y*4 + x), G: 20, B: 40, A: 255})
		}
	}
	processed, err := PreprocessGemma4Image(input, spec)
	if err != nil {
		t.Fatal(err)
	}
	if processed.GridH != 2 || processed.GridW != 2 || len(processed.PixelValues) != 48 {
		t.Fatalf("processed = grid %dx%d values %d", processed.GridH, processed.GridW, len(processed.PixelValues))
	}
	want := []float32{0, 20.0 / 255, 40.0 / 255, 1.0 / 255, 20.0 / 255, 40.0 / 255}
	for index, value := range want {
		if processed.PixelValues[index] != value {
			t.Fatalf("pixel[%d] = %g, want %g", index, processed.PixelValues[index], value)
		}
	}
	wantPositions := []int{0, 0, 1, 0, 0, 1, 1, 1}
	for index, value := range wantPositions {
		if processed.Positions[index] != value {
			t.Fatalf("position[%d] = %d, want %d", index, processed.Positions[index], value)
		}
	}
}

func TestGemma4RealFixture(t *testing.T) {
	projectorPath := os.Getenv("LLAMACPP2GO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("LLAMACPP2GO_GEMMA4_IMAGE")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_GOLDEN")
	if projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_MMPROJ, LLAMACPP2GO_GEMMA4_IMAGE, and LLAMACPP2GO_GEMMA4_GOLDEN")
	}
	type probeRecord struct {
		Shape      []int     `json:"shape"`
		ProbeIndex []int     `json:"probe_index"`
		ProbeValue []float32 `json:"probe_value"`
		L2         float64   `json:"l2"`
	}
	var golden struct {
		ImageHW                   []int `json:"image_hw"`
		NumImagePlaceholderTokens int   `json:"num_image_placeholder_tokens"`
		Preprocess                struct {
			PixelValues probeRecord `json:"pixel_values"`
			Positions   [][]int     `json:"image_position_ids"`
		} `json:"preprocess"`
		ProjectorIntermediates struct {
			PatchLN1            probeRecord `json:"patch_ln1"`
			PatchDense          probeRecord `json:"patch_dense"`
			PatchLN2            probeRecord `json:"patch_ln2"`
			PosNorm             probeRecord `json:"pos_norm"`
			PreProjectionNorm   probeRecord `json:"pre_projection_norm"`
			EmbeddingProjection probeRecord `json:"embedding_projection"`
		} `json:"projector_intermediates"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
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
	runner, err := OpenGemma4(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	processed, err := PreprocessGemma4Image(input, runner.Spec())
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.ImageHW) != 2 || processed.GridH != golden.ImageHW[0]/runner.Spec().ModelPatch ||
		processed.GridW != golden.ImageHW[1]/runner.Spec().ModelPatch {
		t.Fatalf("grid = %dx%d for image %v", processed.GridH, processed.GridW, golden.ImageHW)
	}
	if len(processed.PixelValues) != golden.Preprocess.PixelValues.Shape[1]*golden.Preprocess.PixelValues.Shape[2] {
		t.Fatalf("pixel values = %d, want shape %v", len(processed.PixelValues), golden.Preprocess.PixelValues.Shape)
	}
	compareProbes(t, "Gemma 4 pixel values", processed.PixelValues, golden.Preprocess.PixelValues, 1e-6)
	if len(golden.Preprocess.Positions) != len(processed.Positions)/2 {
		t.Fatalf("positions = %d, want %d", len(processed.Positions)/2, len(golden.Preprocess.Positions))
	}
	for row, position := range golden.Preprocess.Positions {
		if len(position) != 2 || processed.Positions[row*2] != position[0] || processed.Positions[row*2+1] != position[1] {
			t.Fatalf("position %d = %v, want %v", row, processed.Positions[row*2:row*2+2], position)
		}
	}
	if os.Getenv("LLAMACPP2GO_GEMMA4_PROJECTOR_FULL") == "" {
		return
	}
	stages := map[string]probeRecord{
		"patch_ln1":            golden.ProjectorIntermediates.PatchLN1,
		"patch_dense":          golden.ProjectorIntermediates.PatchDense,
		"patch_ln2":            golden.ProjectorIntermediates.PatchLN2,
		"pos_norm":             golden.ProjectorIntermediates.PosNorm,
		"pre_projection_norm":  golden.ProjectorIntermediates.PreProjectionNorm,
		"embedding_projection": golden.ProjectorIntermediates.EmbeddingProjection,
	}
	output, err := runner.encodeWithTrace(context.Background(), processed, func(name string, values []float32) {
		if name == "patch_ln1" {
			values = gemma4InterleavePatchRows(values, runner.Spec().PatchWidth)
		}
		compareGemma4StageProbes(t, "Gemma 4 "+name, values, stages[name])
	})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(uint64(runner.Spec().Hidden), uint64(golden.NumImagePlaceholderTokens))) {
		t.Fatalf("embedding shape = %v", output.Embeddings.Shape)
	}
	compareProbes(t, "Gemma 4 embeddings", output.Embeddings.Data, golden.ProjectorIntermediates.EmbeddingProjection, 0.08)
	var sumSquares float64
	for _, value := range output.Embeddings.Data {
		sumSquares += float64(value) * float64(value)
	}
	l2 := math.Sqrt(sumSquares)
	if relative := math.Abs(l2-golden.ProjectorIntermediates.EmbeddingProjection.L2) / golden.ProjectorIntermediates.EmbeddingProjection.L2; relative > 0.02 {
		t.Fatalf("embedding L2 = %g, want %g (relative %g)", l2, golden.ProjectorIntermediates.EmbeddingProjection.L2, relative)
	}
}

func compareGemma4StageProbes(t *testing.T, name string, values []float32, record struct {
	Shape      []int     `json:"shape"`
	ProbeIndex []int     `json:"probe_index"`
	ProbeValue []float32 `json:"probe_value"`
	L2         float64   `json:"l2"`
}) {
	t.Helper()
	if len(record.ProbeIndex) != len(record.ProbeValue) {
		t.Fatalf("%s probe record is inconsistent", name)
	}
	for index, flat := range record.ProbeIndex {
		if flat < 0 || flat >= len(values) {
			t.Fatalf("%s probe index %d is out of range", name, flat)
		}
		want := record.ProbeValue[index]
		tolerance := max(float32(0.25), float32(math.Abs(float64(want)))*0.1)
		if delta := float32(math.Abs(float64(values[flat] - want))); delta > tolerance {
			t.Fatalf("%s[%d] = %g, want %g (delta %g, tolerance %g)", name, flat, values[flat], want, delta, tolerance)
		}
	}
}

func gemma4InterleavePatchRows(values []float32, width int) []float32 {
	if width <= 0 || width%3 != 0 || len(values)%width != 0 {
		return values
	}
	result := make([]float32, len(values))
	area := width / 3
	for row := 0; row < len(values)/width; row++ {
		for pixel := 0; pixel < area; pixel++ {
			for channel := 0; channel < 3; channel++ {
				result[row*width+pixel*3+channel] = values[row*width+channel*area+pixel]
			}
		}
	}
	return result
}

func TestGemma4RealPromptTokens(t *testing.T) {
	vocabPath := os.Getenv("LLAMACPP2GO_GEMMA4_VOCAB")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_GOLDEN")
	if vocabPath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_VOCAB and LLAMACPP2GO_GEMMA4_GOLDEN")
	}
	var golden struct {
		InputIDs                  []tokenizer.TokenID `json:"input_ids"`
		NumImagePlaceholderTokens int                 `json:"num_image_placeholder_tokens"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(vocabPath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	text := Gemma4ImagePromptText("What color dominates this image? One word.", golden.NumImagePlaceholderTokens)
	ids, err := vocab.Encode(text, tokenizer.EncodeOptions{ParseSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(golden.InputIDs) {
		t.Fatalf("prompt tokens = %d, want %d", len(ids), len(golden.InputIDs))
	}
	for index, id := range ids {
		if id != golden.InputIDs[index] {
			t.Fatalf("prompt token %d = %d, want %d", index, id, golden.InputIDs[index])
		}
	}
}

func TestGemma4RealAudioFixture(t *testing.T) {
	projectorPath := os.Getenv("LLAMACPP2GO_GEMMA4_MMPROJ")
	wavePath := os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_WAVE")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	if projectorPath == "" || wavePath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_MMPROJ, LLAMACPP2GO_GEMMA4_AUDIO_WAVE, and LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	}
	type probeRecord struct {
		Shape      []int     `json:"shape"`
		ProbeIndex []int     `json:"probe_index"`
		ProbeValue []float32 `json:"probe_value"`
		L2         float64   `json:"l2"`
	}
	var golden struct {
		SamplesPerToken    int                    `json:"samples_per_token"`
		NumAudioTokens     int                    `json:"num_audio_placeholder_tokens"`
		InputFeatures      probeRecord            `json:"input_features"`
		AudioIntermediates map[string]probeRecord `json:"audio_intermediates"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("wave byte count = %d", len(raw))
	}
	samples := make([]float32, len(raw)/4)
	for index := range samples {
		samples[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	runner, err := OpenGemma4(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	spec, err := runner.AudioSpec()
	if err != nil {
		t.Fatal(err)
	}
	frames, rows, err := PreprocessGemma4Audio(samples, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.SamplesPerToken != golden.SamplesPerToken || rows != golden.NumAudioTokens {
		t.Fatalf("audio spec=%+v rows=%d, want width=%d rows=%d", spec, rows, golden.SamplesPerToken, golden.NumAudioTokens)
	}
	compareProbes(t, "Gemma 4 audio frames", frames, golden.InputFeatures, 1e-6)
	output, err := runner.EncodeAudio(context.Background(), samples)
	if err != nil {
		t.Fatal(err)
	}
	if !output.Embeddings.Shape.Equal(mustShape(uint64(spec.Hidden), uint64(rows))) {
		t.Fatalf("audio embedding shape = %v", output.Embeddings.Shape)
	}
	soft := golden.AudioIntermediates["audio_soft_tokens"]
	compareProbes(t, "Gemma 4 audio embeddings", output.Embeddings.Data, soft, 0.03)
	var sumSquares float64
	for _, value := range output.Embeddings.Data {
		sumSquares += float64(value) * float64(value)
	}
	if relative := math.Abs(math.Sqrt(sumSquares)-soft.L2) / soft.L2; relative > 0.02 {
		t.Fatalf("audio embedding L2 relative = %g", relative)
	}
}

func TestGemma4RealAudioPromptTokens(t *testing.T) {
	vocabPath := os.Getenv("LLAMACPP2GO_GEMMA4_VOCAB")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	if vocabPath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_VOCAB and LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	}
	var golden struct {
		InputIDs       []tokenizer.TokenID `json:"input_ids"`
		NumAudioTokens int                 `json:"num_audio_placeholder_tokens"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(vocabPath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	text := Gemma4AudioPromptText("What note do you hear? One word.", golden.NumAudioTokens)
	ids, err := vocab.Encode(text, tokenizer.EncodeOptions{ParseSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(golden.InputIDs) {
		t.Fatalf("audio prompt tokens = %d, want %d", len(ids), len(golden.InputIDs))
	}
	for index, id := range ids {
		if id != golden.InputIDs[index] {
			t.Fatalf("audio prompt token %d = %d, want %d", index, id, golden.InputIDs[index])
		}
	}
}

func TestGemma4RealResizeFixture(t *testing.T) {
	projectorPath := os.Getenv("LLAMACPP2GO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("LLAMACPP2GO_GEMMA4_RESIZE_IMAGE")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_RESIZE_GOLDEN")
	if projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_MMPROJ, LLAMACPP2GO_GEMMA4_RESIZE_IMAGE, and LLAMACPP2GO_GEMMA4_RESIZE_GOLDEN")
	}
	type probeRecord struct {
		ProbeIndex []int     `json:"probe_index"`
		ProbeValue []float32 `json:"probe_value"`
	}
	var golden struct {
		ResizeOutHW []int       `json:"resize_out_hw"`
		NumSoft     int         `json:"num_soft_tokens"`
		PixelValues probeRecord `json:"pixel_values"`
		Positions   [][]int     `json:"image_position_ids"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
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
	runner, err := OpenGemma4(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	processed, err := PreprocessGemma4Image(input, runner.Spec())
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.ResizeOutHW) != 2 || processed.GridH*runner.Spec().ModelPatch != golden.ResizeOutHW[0] ||
		processed.GridW*runner.Spec().ModelPatch != golden.ResizeOutHW[1] || processed.GridH*processed.GridW != golden.NumSoft {
		t.Fatalf("resize = %dx%d grid %dx%d, want %v and %d tokens",
			processed.GridW*runner.Spec().ModelPatch, processed.GridH*runner.Spec().ModelPatch,
			processed.GridW, processed.GridH, golden.ResizeOutHW, golden.NumSoft)
	}
	for probe, offset := range golden.PixelValues.ProbeIndex {
		if offset >= len(processed.PixelValues) {
			continue
		}
		if delta := float32(math.Abs(float64(processed.PixelValues[offset] - golden.PixelValues.ProbeValue[probe]))); delta > 2.0/255 {
			t.Fatalf("resized pixel[%d] = %g, want %g", offset, processed.PixelValues[offset], golden.PixelValues.ProbeValue[probe])
		}
	}
	for row := 0; row < golden.NumSoft; row++ {
		position := golden.Positions[row]
		if len(position) != 2 || processed.Positions[row*2] != position[0] || processed.Positions[row*2+1] != position[1] {
			t.Fatalf("resized position %d = %v, want %v", row, processed.Positions[row*2:row*2+2], position)
		}
	}
}

func tinyGemma4Metadata() []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}},
		{Key: "clip.vision.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: gemma4UVProjectorType}},
		{Key: "clip.has_vision_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.vision.patch_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(1)}},
		{Key: "clip.vision.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.vision.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
		{Key: "clip.audio.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: gemma4UAProjectorType}},
		{Key: "clip.has_audio_encoder", Value: gguf.Value{Type: gguf.ValueTypeBool, Data: true}},
		{Key: "clip.audio.embedding_length", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(3)}},
		{Key: "clip.audio.projection_dim", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "clip.audio.attention.layer_norm_epsilon", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e-6)}},
	}
}

func tinyGemma4Tensors() []gguf.TensorData {
	return []gguf.TensorData{
		f32Tensor("v.patch_embd.weight", []uint64{3, 2}, nil),
		f32Tensor("v.patch_embd.bias", []uint64{2}, nil),
		f32Tensor("v.patch_norm.1.weight", []uint64{3}, []float32{1, 1, 1}),
		f32Tensor("v.patch_norm.1.bias", []uint64{3}, nil),
		f32Tensor("v.patch_norm.2.weight", []uint64{2}, []float32{1, 1}),
		f32Tensor("v.patch_norm.2.bias", []uint64{2}, nil),
		f32Tensor("v.position_embd.weight", []uint64{2, 32, 2}, nil),
		f32Tensor("v.patch_norm.3.weight", []uint64{2}, []float32{1, 1}),
		f32Tensor("v.patch_norm.3.bias", []uint64{2}, nil),
		f32Tensor("mm.input_projection.weight", []uint64{2, 2}, nil),
		f32Tensor("mm.a.input_projection.weight", []uint64{3, 2}, []float32{1, 0, 0, 0, 1, 0}),
	}
}
