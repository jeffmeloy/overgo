package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/vitencoder"
)

const (
	// captureFrameFPS and captureFrameCount cut a clip into the few frames the frame suite cuts, whose middle one a
	// sample encodes, as its cases' frames are.
	captureFrameFPS   = 1.0
	captureFrameCount = 9
	// captureEmbeddings is the text model's token embedding table a caption's tokens are read from.
	captureEmbeddings = "model.embed_tokens.weight"
	// captureCLS and capturePatchMean name the encoder's two captured outputs: its class token and the mean of its
	// patch tokens.
	captureCLS       = sourcePrefix + "dino-cls"
	capturePatchMean = sourcePrefix + "dino-patch-mean"
	// tokensSuffix ends the name of a sample's cached frame tokens.
	tokensSuffix = ".tokens.f32"
	// float32Bytes is the bytes of one cached value, encoderOutputs the encoder's outputs a sample holds before its
	// caption's, and halfway what a clip's frame count is divided by for its middle frame.
	float32Bytes   = 4
	encoderOutputs = 2
	halfway        = 2
)

// captureSpec names a capture: the caption-matching suite its samples stand beside, the encoder and the text model
// whose representations it pairs, how many samples and the seed that picks them, the file it writes, the directory
// each sample is cached in so a long capture resumes, and the digest of the acceptance it serves.
type captureSpec struct {
	Clips      string `json:"clips"`
	Captions   string `json:"captions"`
	Cases      int    `json:"cases"`
	SuiteSeed  uint64 `json:"suite_seed"`
	Encoder    string `json:"encoder"`
	TextModel  string `json:"text_model"`
	Samples    int    `json:"samples"`
	Seed       uint64 `json:"seed"`
	Output     string `json:"output"`
	Cache      string `json:"cache"`
	Acceptance string `json:"acceptance"`
}

// captureSamples are the clips a capture reads: the training side of the suite's split, shuffled by the capture's seed,
// as many as it asks or as the side holds.
func captureSamples(spec captureSpec) ([]evaluation.CaptionTrainingClip, error) {
	clips, err := evaluation.CaptionTrainingClips(evaluation.CaptionMatchingSpec{
		Clips: spec.Clips, Captions: spec.Captions, Cases: spec.Cases, Seed: spec.SuiteSeed,
	})
	if err != nil {
		return nil, err
	}
	draws := rand.New(rand.NewPCG(spec.Seed, spec.Seed))
	draws.Shuffle(len(clips), func(left, right int) { clips[left], clips[right] = clips[right], clips[left] })
	return clips[:min(spec.Samples, len(clips))], nil
}

// captureSeams writes a capture of the encoder's class token and patch mean for each sample's middle frame beside its
// caption's mean token embedding in the text model, each sample grouped by its source video.
func captureSeams(ctx context.Context, specPath string) error {
	var spec captureSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	if spec.Samples <= 0 || spec.Output == "" || spec.Cache == "" || spec.Acceptance == "" {
		return errors.New("seam-align -capture names its samples, its output, its cache and the acceptance it serves")
	}
	samples, err := captureSamples(spec)
	if err != nil {
		return err
	}
	encoder, err := vitencoder.Load(spec.Encoder)
	if err != nil {
		return err
	}
	tokenizer, err := hfbpe.Load(spec.TextModel)
	if err != nil {
		return err
	}
	text, err := safetensors.OpenSource(spec.TextModel)
	if err != nil {
		return err
	}
	table, found := text.Tensors[captureEmbeddings]
	_ = text.Close()
	if !found || len(table.Shape) != matrixRank {
		return fmt.Errorf("seam-align: %s holds no %s table", spec.TextModel, captureEmbeddings)
	}
	_, _, hidden := encoder.Shape()
	width := int(table.Shape[1])
	ffmpeg, err := media.ResolveFFmpeg(os.Getenv("OVERGO_FFMPEG"))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(spec.Cache, clioptions.OutputDirectoryMode); err != nil {
		return err
	}
	cls, patches, target := make([]float32, 0, len(samples)*hidden), make([]float32, 0, len(samples)*hidden), make([]float32, 0, len(samples)*width)
	groups, sources := make([]float32, len(samples)), map[string]int{}
	for index, sample := range samples {
		values, err := capturedSample(ctx, spec, encoder, tokenizer, ffmpeg, sample, hidden, width)
		if err != nil {
			return fmt.Errorf("seam-align: %s: %w", sample.Vid, err)
		}
		cls, patches, target = append(cls, values[:hidden]...), append(patches, values[hidden:encoderOutputs*hidden]...), append(target, values[encoderOutputs*hidden:]...)
		if _, seen := sources[sample.Source]; !seen {
			sources[sample.Source] = len(sources)
		}
		groups[index] = float32(sources[sample.Source])
		fmt.Fprintf(os.Stderr, "seam-align: captured %d of %d\n", index+1, len(samples))
	}
	count := len(samples)
	return safetensors.Save(spec.Output,
		map[string][]float32{captureCLS: cls, capturePatchMean: patches, targetTensor: target, groupsTensor: groups},
		map[string][]int{captureCLS: {count, hidden}, capturePatchMean: {count, hidden}, targetTensor: {count, width}, groupsTensor: {count}},
		map[string]string{"encoder": spec.Encoder, "text_model": spec.TextModel, "acceptance": spec.Acceptance, "samples": strconv.Itoa(count)})
}

// capturedSample is one sample's representations, the encoder's class token and patch mean, then the caption's mean
// token embedding, read from the cache or captured and cached. Every token of the frame is cached beside them, the
// encoder's width by its tokens, for the bridge's training, which reads them all.
func capturedSample(
	ctx context.Context, spec captureSpec, encoder *vitencoder.Encoder, tokenizer *hfbpe.Tokenizer, ffmpeg string,
	sample evaluation.CaptionTrainingClip, hidden, width int,
) ([]float32, error) {
	cached, frameTokens := filepath.Join(spec.Cache, sample.Vid+".f32"), filepath.Join(spec.Cache, sample.Vid+tokensSuffix)
	if values, err := readFloats(cached); err == nil && len(values) == encoderOutputs*hidden+width {
		if _, err := os.Stat(frameTokens); err == nil {
			return values, nil
		}
	}
	frames, err := media.DecodeVideoFile(ctx, filepath.Join(spec.Clips, sample.Vid+".mp4"), ffmpeg, captureFrameFPS, captureFrameCount)
	if err != nil || len(frames) == 0 {
		return nil, errors.Join(err, errors.New("no frame to cut"))
	}
	var frame bytes.Buffer
	if err := png.Encode(&frame, frames[len(frames)/halfway]); err != nil {
		return nil, err
	}
	tokens, err := encoder.Tokens(frame.Bytes())
	if err != nil {
		return nil, err
	}
	if err := writeFloats(frameTokens, tokens); err != nil {
		return nil, err
	}
	values := make([]float32, encoderOutputs*hidden+width)
	copy(values, tokens[:hidden])
	_, grid, _ := encoder.Shape()
	patches := grid * grid
	first := len(tokens)/hidden - patches
	for patch := range patches {
		for channel := range hidden {
			values[hidden+channel] += tokens[(first+patch)*hidden+channel] / float32(patches)
		}
	}
	ids, err := tokenizer.Encode(sample.Caption)
	if err != nil || len(ids) == 0 {
		return nil, errors.Join(err, errors.New("the caption gives no tokens"))
	}
	rows, err := safetensors.ReadTensorRowsF32(spec.TextModel, captureEmbeddings, width, ids)
	if err != nil {
		return nil, err
	}
	for row := range len(ids) {
		for column := range width {
			values[encoderOutputs*hidden+column] += rows[row*width+column] / float32(len(ids))
		}
	}
	return values, writeFloats(cached, values)
}

// readFloats reads a cache file of little-endian float32 values.
func readFloats(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	values := make([]float32, len(data)/float32Bytes)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*float32Bytes:]))
	}
	return values, nil
}

// writeFloats writes values to a cache file as little-endian float32.
func writeFloats(path string, values []float32) error {
	data := make([]byte, len(values)*float32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*float32Bytes:], math.Float32bits(value))
	}
	return os.WriteFile(path, data, clioptions.PrivateFileMode)
}
