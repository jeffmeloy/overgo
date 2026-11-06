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
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/modelartifact"
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
	// tokensSuffix ends the name of a sample's cached frame tokens, and samplesSuffix the name of the file beside the
	// capture that names its samples for the bridge's training.
	tokensSuffix  = ".tokens.f32"
	samplesSuffix = ".samples.json"
	// bindingSuffix ends the name of a sample's cached representations and what they were made from.
	bindingSuffix = ".binding.json"
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
// caption's mean token embedding in the text model, each sample grouped by its source video, and beside the capture
// the samples it read, with where their frame tokens are cached, as the bridge's training reads them.
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
	inputs, err := captureInputs(spec, ffmpeg)
	if err != nil {
		return err
	}
	captured := make([]evaluation.CapturedCaption, len(samples))
	cls, patches, target := make([]float32, 0, len(samples)*hidden), make([]float32, 0, len(samples)*hidden), make([]float32, 0, len(samples)*width)
	groups, sources := make([]float32, len(samples)), map[string]int{}
	for index, sample := range samples {
		values, frame, err := capturedSample(ctx, spec, inputs, encoder, tokenizer, ffmpeg, sample, hidden, width)
		if err != nil {
			return fmt.Errorf("seam-align: %s: %w", sample.Vid, err)
		}
		captured[index] = evaluation.CapturedCaption{CaptionTrainingClip: sample, Frame: frame}
		cls, patches, target = append(cls, values[:hidden]...), append(patches, values[hidden:encoderOutputs*hidden]...), append(target, values[encoderOutputs*hidden:]...)
		if _, seen := sources[sample.Source]; !seen {
			sources[sample.Source] = len(sources)
		}
		groups[index] = float32(sources[sample.Source])
		fmt.Fprintf(os.Stderr, "seam-align: captured %d of %d\n", index+1, len(samples))
	}
	count := len(samples)
	if err := jsonfile.Write(spec.Output+samplesSuffix, evaluation.CaptionTrainingCapture{
		Encoder: spec.Encoder, TextModel: spec.TextModel, Acceptance: spec.Acceptance, Tokens: spec.Cache, EncoderWidth: hidden, TextWidth: width, Samples: captured,
	}, clioptions.PrivateFileMode); err != nil {
		return err
	}
	return safetensors.Save(spec.Output,
		map[string][]float32{captureCLS: cls, capturePatchMean: patches, targetTensor: target, groupsTensor: groups},
		map[string][]int{captureCLS: {count, hidden}, capturePatchMean: {count, hidden}, targetTensor: {count, width}, groupsTensor: {count}},
		map[string]string{"encoder": spec.Encoder, "text_model": spec.TextModel, "acceptance": spec.Acceptance, "samples": strconv.Itoa(count)})
}

// sampleBinding is a cached sample: what it was made from, its representations, and the identity of the frame tokens
// kept beside it.
type sampleBinding struct {
	Inputs string      `json:"inputs"`
	Values []float32   `json:"values"`
	Frame  artifact.ID `json:"frame"`
}

// captureInputs is the identity of what every sample's representations are made from: this capture's own executable,
// whose bytes hold its code and how it cuts and reads a sample, the encoder's and the text model's files, and the
// decoder that cuts the frames.
func captureInputs(spec captureSpec, ffmpeg string) (string, error) {
	executable, err := os.Executable()
	if err != nil {
		return "", err
	}
	var identities []string
	for _, path := range []string{executable, ffmpeg} {
		identity, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			return "", err
		}
		identities = append(identities, identity.String())
	}
	for _, model := range []string{spec.Encoder, spec.TextModel} {
		inventory, err := modelartifact.FromHFPath(model)
		if err != nil {
			return "", err
		}
		identities = append(identities, inventory.Manifest.ID.String())
	}
	return strings.Join(identities, "\n"), nil
}

// sampleKey is the identity a sample's cached representations are kept under: the capture's inputs, the bytes of the
// sample's clip and its caption.
func sampleKey(inputs, clip, caption string) (string, error) {
	identity, err := artifact.IdentifyFile(artifact.KindFile, clip)
	return strings.Join([]string{inputs, identity.String(), caption}, "\n"), err
}

// cachedSample is a sample's count representations and its frame tokens' identity as the cache keeps them, when it
// keeps them under key and its frame tokens are the bytes it recorded.
func cachedSample(cache, vid, key string, count int) ([]float32, artifact.ID, bool) {
	var binding sampleBinding
	if jsonfile.DecodeStrict(filepath.Join(cache, vid+bindingSuffix), &binding) != nil || binding.Inputs != key || len(binding.Values) != count {
		return nil, artifact.ID{}, false
	}
	frame, err := artifact.IdentifyFile(artifact.KindFile, filepath.Join(cache, vid+tokensSuffix))
	return binding.Values, frame, err == nil && frame == binding.Frame
}

// keepSample caches a sample's representations and its frame tokens under key, and returns the tokens' identity. The
// binding is written after the tokens, so a sample cut short in its writing is no hit.
func keepSample(cache, vid, key string, values, tokens []float32) (artifact.ID, error) {
	frames := filepath.Join(cache, vid+tokensSuffix)
	if err := writeFloats(frames, tokens); err != nil {
		return artifact.ID{}, err
	}
	frame, err := artifact.IdentifyFile(artifact.KindFile, frames)
	if err != nil {
		return artifact.ID{}, err
	}
	return frame, jsonfile.Write(filepath.Join(cache, vid+bindingSuffix), sampleBinding{Inputs: key, Values: values, Frame: frame}, clioptions.PrivateFileMode)
}

// capturedSample is one sample's representations, the encoder's class token and patch mean, then the caption's mean
// token embedding, and the identity of its frame's tokens, read from the cache when it keeps them under the sample's
// inputs or captured and cached. Every token of the frame is cached beside them, the encoder's width by its tokens,
// for the bridge's training, which reads them all.
func capturedSample(
	ctx context.Context, spec captureSpec, inputs string, encoder *vitencoder.Encoder, tokenizer *hfbpe.Tokenizer, ffmpeg string,
	sample evaluation.CaptionTrainingClip, hidden, width int,
) (values []float32, identity artifact.ID, err error) {
	clip := filepath.Join(spec.Clips, sample.Vid+".mp4")
	key, err := sampleKey(inputs, clip, sample.Caption)
	if err != nil {
		return nil, identity, err
	}
	if values, frame, kept := cachedSample(spec.Cache, sample.Vid, key, encoderOutputs*hidden+width); kept {
		return values, frame, nil
	}
	frames, err := media.DecodeVideoFile(ctx, clip, ffmpeg, captureFrameFPS, captureFrameCount)
	if err != nil || len(frames) == 0 {
		return nil, identity, errors.Join(err, errors.New("no frame to cut"))
	}
	var frame bytes.Buffer
	if err := png.Encode(&frame, frames[len(frames)/halfway]); err != nil {
		return nil, identity, err
	}
	tokens, err := encoder.Tokens(frame.Bytes())
	if err != nil {
		return nil, identity, err
	}
	values = make([]float32, encoderOutputs*hidden+width)
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
		return nil, identity, errors.Join(err, errors.New("the caption gives no tokens"))
	}
	rows, err := safetensors.ReadTensorRowsF32(spec.TextModel, captureEmbeddings, width, ids)
	if err != nil {
		return nil, identity, err
	}
	for row := range len(ids) {
		for column := range width {
			values[encoderOutputs*hidden+column] += rows[row*width+column] / float32(len(ids))
		}
	}
	identity, err = keepSample(spec.Cache, sample.Vid, key, values, tokens)
	return values, identity, err
}

// writeFloats writes values to a cache file as little-endian float32.
func writeFloats(path string, values []float32) error {
	data := make([]byte, len(values)*float32Bytes)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*float32Bytes:], math.Float32bits(value))
	}
	return os.WriteFile(path, data, clioptions.PrivateFileMode)
}
