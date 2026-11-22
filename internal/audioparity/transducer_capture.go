package audioparity

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/gitauthority"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

type transducerCapture struct {
	Version              int               `json:"version"`
	Model                artifact.ID       `json:"model"`
	Revision             string            `json:"revision"`
	License              string            `json:"license"`
	Files                map[string]string `json:"files"`
	Fixture              string            `json:"fixture"`
	Reference            string            `json:"reference"`
	AudioSHA256          string            `json:"audio_sha256"`
	Samples              int               `json:"samples"`
	SampleRate           int               `json:"sample_rate"`
	PromptID             int               `json:"prompt_id"`
	Lookahead            int               `json:"lookahead"`
	FirstChunkFrames     int               `json:"first_chunk_frames"`
	FollowingChunkFrames int               `json:"following_chunk_frames"`
	Dataset              struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
		Row    int    `json:"row"`
	} `json:"dataset"`
	Runtime struct {
		Python       string            `json:"python"`
		Torch        string            `json:"torch"`
		Transformers string            `json:"transformers"`
		Librosa      string            `json:"librosa"`
		Tokenizers   string            `json:"tokenizers"`
		Threads      int               `json:"threads"`
		Device       string            `json:"device"`
		Attention    string            `json:"attention"`
		Sources      map[string]string `json:"sources"`
	} `json:"runtime,omitzero"`
	// Producer names overgo as the capture's maker where overgo remade it (cmd/speech-capture); a capture the
	// reference software made names its runtime instead.
	Producer      captureProducer            `json:"producer,omitzero"`
	CaptureSHA256 string                     `json:"capture_sha256"`
	TensorsSHA256 string                     `json:"tensors_sha256"`
	Captures      map[string]transducerTrace `json:"captures"`
	TensorCount   int                        `json:"tensor_count"`
	Nonfinite     map[string]int             `json:"nonfinite"`
}

// transducerTrace is one mode's run of a transducer capture: the mel chunks it was fed, the token each decoder step
// chose, its wall time and the text its emissions decode to.
type transducerTrace struct {
	Chunks []transducerChunk `json:"chunks"`
	Steps  []int             `json:"steps"`
	WallNS uint64            `json:"wall_ns"`
	Text   string            `json:"text"`
}

// transducerChunk is one chunk of mel frames a transducer trace was fed.
type transducerChunk struct {
	Frames int `json:"frames"`
}

var transducerCaptureContract = artifact.JSONContract(artifact.KindEvidence, "overgo/audio-transducer-golden/v1")

func (c transducerCapture) validate() error {
	if c.Version != 1 || !c.Model.Valid() || c.Model.Kind() != artifact.KindModel || !gitauthority.ValidObjectID(c.Revision) || strings.ToLower(c.Revision) != c.Revision || c.License == "" || c.Fixture == "" || c.Reference == "" ||
		c.Samples <= 0 || c.SampleRate <= 0 || c.PromptID < 0 || c.Lookahead < 0 || c.FirstChunkFrames <= 0 || c.FollowingChunkFrames <= 0 || c.Dataset.Row < 0 || c.Dataset.Path == "" ||
		c.TensorCount <= 0 || len(c.Nonfinite) != 0 || len(c.Captures) != 2 || len(c.Files) == 0 {
		return errors.New("transducer capture: incomplete source, runtime, corpus or finite trace declaration")
	}
	reference := c.Runtime.Device == "cpu" && slices.Contains([]string{"eager", "sdpa"}, c.Runtime.Attention) && c.Runtime.Threads > 0 && c.Runtime.Python != "" && c.Runtime.Torch != "" &&
		c.Runtime.Transformers != "" && c.Runtime.Librosa != "" && c.Runtime.Tokenizers != "" && len(c.Runtime.Sources) != 0
	if err := captureMaker(c.Producer, c.Runtime, reference); err != nil {
		return fmt.Errorf("transducer capture: incomplete source, runtime, corpus or finite trace declaration: %w", err)
	}
	for _, paths := range []map[string]string{c.Files, c.Runtime.Sources, {c.Dataset.Path: c.Dataset.SHA256}} {
		for name := range paths {
			if err := validateRelativePath(name); err != nil || name == "." || name == ".." || strings.Contains(name, ":") {
				return fmt.Errorf("transducer capture: invalid member path %q", name)
			}
		}
	}
	for _, digest := range append([]string{c.AudioSHA256, c.Dataset.SHA256, c.CaptureSHA256, c.TensorsSHA256}, mapDigests(c.Files, c.Runtime.Sources)...) {
		if _, err := artifact.ParseID("file:sha256:" + digest); err != nil {
			return err
		}
	}
	for _, mode := range []string{"offline", "streaming"} {
		trace, ok := c.Captures[mode]
		if !ok || len(trace.Chunks) == 0 || len(trace.Steps) == 0 || trace.WallNS == 0 || trace.Text == "" {
			return fmt.Errorf("transducer capture: incomplete %s trace", mode)
		}
		for _, chunk := range trace.Chunks {
			if chunk.Frames <= 0 {
				return errors.New("transducer capture: invalid chunk")
			}
		}
		for _, token := range trace.Steps {
			if token < 0 {
				return errors.New("transducer capture: negative token")
			}
		}
	}
	return nil
}

func mapDigests(groups ...map[string]string) []string {
	var values []string
	for _, m := range groups {
		values = slices.AppendSeq(values, maps.Values(m))
	}
	return values
}

// PublishTransducerCapture uses the existing external-tensor publication owner.
// It admits source-bound native traces, not a Go parity or quality claim.
func PublishTransducerCapture(ctx context.Context, repository artifact.Repository, reportPath, sourcePath, tensorsPath string) (artifact.ID, error) {
	if ctx == nil || repository == nil {
		return artifact.ID{}, errors.New("transducer capture: incomplete invocation")
	}
	if err := ctx.Err(); err != nil {
		return artifact.ID{}, err
	}
	data, err := artifact.ReadContentFile(reportPath)
	if err != nil {
		return artifact.ID{}, err
	}
	var capture transducerCapture
	if err := strictjson.DecodeBytes(data, &capture); err != nil {
		return artifact.ID{}, err
	}
	source, err := artifact.ReadContentFile(sourcePath)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishTransducerCapture(ctx, repository, capture, source, tensorsPath)
}

// publishTransducerCapture registers a validated transducer capture of a registered model, every component of which
// its files name, with the source that made it and its tensors file.
func publishTransducerCapture(ctx context.Context, repository artifact.Repository, capture transducerCapture, source []byte, tensorsPath string) (artifact.ID, error) {
	if err := capture.validate(); err != nil {
		return artifact.ID{}, err
	}
	manifest, found, err := repository.Manifest(ctx, capture.Model)
	if err != nil || !found {
		return artifact.ID{}, errors.Join(errors.New("transducer capture: registered model absent"), err)
	}
	if len(manifest.Components) == 0 {
		return artifact.ID{}, errors.New("transducer capture: registered model has no components")
	}
	fileDigests := mapDigests(capture.Files)
	for _, component := range manifest.Components {
		if !slices.Contains(fileDigests, component.Artifact.DigestHex()) {
			return artifact.ID{}, errors.New("transducer capture: model component identity differs")
		}
	}
	golden, err := artifact.JSONContent(transducerCaptureContract, capture)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishTensorCapture(ctx, repository, golden, capture.Model, capture.CaptureSHA256, source, capture.TensorsSHA256, capture.TensorCount, tensorsPath)
}

// RemakeTransducerCapture remakes the transducer capture the store records as id, one of those listPath lists, from
// overgo's own speech path: it runs the capture's recording through the decoder, frontend and transducer offline and
// in the capture's mel chunks as TestSecondASROfflineAcceptance and TestSecondASRStreamingAcceptance run them, keeps
// every boundary those acceptances compare in one tensors file under the references, registers the file where the
// store finds it, publishes the capture naming overgo at the remake's commit as its maker, and lists it in listPath in
// place of id. It returns the published capture's identity.
func RemakeTransducerCapture(ctx context.Context, remake SpeechCaptureRemake, listPath string, id artifact.ID) (artifact.ID, error) {
	if err := remake.ready(ctx); err != nil {
		return artifact.ID{}, err
	}
	// The list names the capture before anything is computed, so a mistaken id costs no model run.
	if _, _, err := listedCapture(listPath, id); err != nil {
		return artifact.ID{}, err
	}
	previous, err := readTransducerCapture(ctx, remake.Store, id)
	if err != nil {
		return artifact.ID{}, err
	}
	directory, err := remake.directory("transducer-" + previous.Fixture)
	if err != nil {
		return artifact.ID{}, err
	}
	capture, tensors, err := traceTransducerCapture(ctx, remake.Store, remake.Datasets, previous)
	if err != nil {
		return artifact.ID{}, err
	}
	path, err := tensors.write(directory)
	if err != nil {
		return artifact.ID{}, err
	}
	return recordTransducerCapture(ctx, remake.Store, capture, remake.producer(), path, listPath, id)
}

// readTransducerCapture is the valid transducer capture the store records as id.
func readTransducerCapture(ctx context.Context, store artifact.Reader, id artifact.ID) (transducerCapture, error) {
	content, err := artifact.RequireTypedContent(ctx, store, id)
	if err != nil {
		return transducerCapture{}, err
	}
	if content.Descriptor.Schema != transducerCaptureContract.Schema {
		return transducerCapture{}, fmt.Errorf("transducer capture: %s is no transducer capture", id)
	}
	var capture transducerCapture
	if err := strictjson.DecodeBytes(content.Data, &capture); err != nil {
		return transducerCapture{}, err
	}
	return capture, capture.validate()
}

// recordTransducerCapture names producer as the maker of capture, whose tensors lie at tensorsPath, publishes it, and
// lists it in listPath in place of replaced.
func recordTransducerCapture(ctx context.Context, repository artifact.Repository, capture transducerCapture, producer captureProducer, tensorsPath, listPath string, replaced artifact.ID) (artifact.ID, error) {
	listed, index, err := listedCapture(listPath, replaced)
	if err != nil {
		return artifact.ID{}, err
	}
	source, sourceSHA, tensorSHA, count, err := producer.recorded(tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	capture.Producer, capture.CaptureSHA256, capture.TensorsSHA256, capture.TensorCount = producer, sourceSHA, tensorSHA, count
	id, err := publishTransducerCapture(ctx, repository, capture, source, tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	listed[index] = id
	return id, jsonfile.Write(listPath, listed, 0o644)
}

// traceTransducerCapture runs previous's recording through overgo's speech path on the CPU, from the model the store
// records as previous's and the corpus shard under datasets, and returns the capture of the same recording, model and
// chunk schedule with every boundary the transducer acceptances compare and no maker named yet.
func traceTransducerCapture(ctx context.Context, store artifact.Reader, datasets string, previous transducerCapture) (transducerCapture, *captureTensors, error) {
	var reviewed recurrentDeclaration
	var encoder speechrecognition.Declaration
	var binding speechrecognition.TransducerBinding
	if err := strictjson.DecodeBytes(transducerExecutionJSON, &reviewed); err != nil {
		return transducerCapture{}, nil, err
	}
	if err := errors.Join(strictjson.DecodeBytes(reviewed.Encoder, &encoder), strictjson.DecodeBytes(reviewed.Transducer, &binding)); err != nil {
		return transducerCapture{}, nil, err
	}
	directory, err := artifact.AvailablePath(ctx, store, previous.Model, artifact.LocationDirectory)
	if err != nil {
		return transducerCapture{}, nil, err
	}
	for name, digest := range previous.Files {
		if err := verifyDigest(directory, name, digest); err != nil {
			return transducerCapture{}, nil, fmt.Errorf("transducer capture: model member %s: %w", name, err)
		}
	}
	pcm, err := transducerRecording(ctx, datasets, previous)
	if err != nil {
		return transducerCapture{}, nil, err
	}
	weights, err := safetensors.OpenSource(directory)
	if err != nil {
		return transducerCapture{}, nil, err
	}
	model, err := speechrecognition.LoadTransducer(ctx, weights, encoder, binding, captureMemoryBytes)
	if err = errors.Join(err, weights.Close()); err != nil {
		return transducerCapture{}, nil, err
	}
	defer model.Close(context.Background())
	frontend, err := audiodsp.NewFrontend(reviewed.Frontend, captureMemoryBytes)
	if err != nil {
		return transducerCapture{}, nil, err
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return transducerCapture{}, nil, err
	}
	path := transducerPath{hop: int(reviewed.Frontend.Geometry.HopSamples), bands: binding.Bands, blocks: len(encoder.Blocks), recurrent: len(binding.Recurrent),
		model: model, decode: tokenizer.DecodeText}
	tensors := &captureTensors{}
	capture := previous
	capture.Runtime = transducerCapture{}.Runtime
	capture.Producer, capture.CaptureSHA256, capture.TensorsSHA256, capture.TensorCount = captureProducer{}, "", "", 0
	capture.Nonfinite = map[string]int{}
	capture.Captures, err = path.trace(ctx, tensors, frontend, pcm, previous)
	return capture, tensors, err
}

// transducerRecording is the decoded recording capture names: row capture.Dataset.Row of the LibriSpeech clean shard
// at capture.Dataset.Path under datasets, its shard and encoded audio identities checked against the capture's.
func transducerRecording(ctx context.Context, datasets string, capture transducerCapture) ([]float32, error) {
	shardRoot := filepath.Join(datasets, "librispeech_asr-clean-xet")
	shard := filepath.Join(shardRoot, filepath.FromSlash(capture.Dataset.Path))
	if err := verifyDigest(shardRoot, capture.Dataset.Path, capture.Dataset.SHA256); err != nil {
		return nil, fmt.Errorf("transducer capture: source shard: %w", err)
	}
	var pcm []float32
	err := dataset.ReadParquetTextRows(ctx, shard, "bytes", capture.Dataset.Row+1, func(row uint64, payload string) error {
		if row != uint64(capture.Dataset.Row) {
			return nil
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, []byte(payload))
		if err != nil || id.DigestHex() != capture.AudioSHA256 {
			return errors.Join(errors.New("transducer capture: source audio changed"), err)
		}
		audio, _, err := media.DecodeAudio(ctx, []byte(payload), uint64(capture.Samples))
		if err != nil {
			return err
		}
		if len(audio.Samples) != capture.Samples || audio.Format.SampleRate != uint64(capture.SampleRate) || audio.Format.Channels != 1 {
			return errors.New("transducer capture: source audio format differs")
		}
		pcm = audio.Samples
		return nil
	})
	if err == nil && pcm == nil {
		err = errors.New("transducer capture: the shard holds no row the capture names")
	}
	return pcm, err
}

// transducerPath is overgo's CPU speech path for a recurrent checkpoint: its frontend's hop in samples, the mel bands
// its transducer reads, its encoder's block count and its decoder's recurrent layer count, the transducer itself and
// its tokenizer's text decoding.
type transducerPath struct {
	hop, bands, blocks, recurrent int
	model                         *speechrecognition.Transducer
	decode                        func([]int) (string, error)
}

// trace keeps the boundaries the transducer acceptances compare for one recording: its waveform, the frontend's
// features and valid-frame mask, then each mode's encoder boundaries per chunk, decoder steps and emissions, offline
// over the whole clip and streaming over the capture's chunk schedule. It returns each mode's trace.
func (p transducerPath) trace(ctx context.Context, tensors *captureTensors, frontend *audiodsp.Frontend, pcm []float32, capture transducerCapture) (map[string]transducerTrace, error) {
	var workspace audiodsp.Workspace
	features, frames, err := frontend.Process(ctx, [][]float32{pcm}, capture.SampleRate, &workspace, audiodsp.ProcessOptions{})
	if err != nil {
		return nil, err
	}
	// The pinned processor zeros the centered STFT's final invalid frame.
	valid, mask := len(pcm)/p.hop, make([]bool, frames)
	for index := range mask {
		mask[index] = index < valid
	}
	bands := p.bands
	tensors.f32("waveform", pcm, len(pcm))
	tensors.mask("mask", mask, frames)
	tensors.f32("features", features, frames, bands)
	traces := map[string]transducerTrace{}
	var offline speechrecognition.TransducerWorkspace
	steps, seen, start := []int{}, 0, time.Now()
	result, err := p.model.Recognize(ctx, features, frames, &offline, p.observer(tensors, "offline", 0, &steps, &seen))
	if err != nil {
		return nil, err
	}
	if seen != p.blocks+1 {
		return nil, errors.New("transducer capture: offline encoder boundary count differs")
	}
	offlineTrace, err := p.emitted(tensors, "offline", result.Tokens, result.Durations, steps, start)
	if err != nil {
		return nil, err
	}
	offlineTrace.Chunks = []transducerChunk{{Frames: frames}}
	traces["offline"] = offlineTrace
	var streaming speechrecognition.TransducerWorkspace
	var tokens, durations []int
	var chunks []transducerChunk
	steps, seen, start = []int{}, 0, time.Now()
	for offset, size := 0, capture.FirstChunkFrames; offset < frames; offset, size = offset+size, capture.FollowingChunkFrames {
		chunk := make([]float32, size*bands)
		copy(chunk, features[offset*bands:min(frames, offset+size)*bands])
		result, err := p.model.RecognizeChunk(ctx, chunk, size, offset+size >= frames, &streaming, p.observer(tensors, "streaming", len(chunks), &steps, &seen))
		if err != nil {
			return nil, err
		}
		tokens, durations = append(tokens, result.Tokens...), append(durations, result.Durations...)
		chunks = append(chunks, transducerChunk{Frames: size})
	}
	if seen != len(chunks)*(p.blocks+1) {
		return nil, errors.New("transducer capture: streaming encoder boundary count differs")
	}
	streamingTrace, err := p.emitted(tensors, "streaming", tokens, durations, steps, start)
	streamingTrace.Chunks = chunks
	traces["streaming"] = streamingTrace
	return traces, err
}

// emitted keeps a mode's emissions and frame advances and returns its trace: the token each decoder step chose, the
// mode's wall time since start and the text its emissions decode to.
func (p transducerPath) emitted(tensors *captureTensors, mode string, tokens, durations, steps []int, start time.Time) (transducerTrace, error) {
	tensors.ids(mode+"/sequences", tokens, 1, len(tokens))
	tensors.ids(mode+"/durations", durations, 1, len(durations))
	text, err := p.decode(tokens)
	return transducerTrace{Steps: steps, WallNS: uint64(time.Since(start)), Text: text}, err
}

// observer keeps every boundary of one call in mode under the names the transducer acceptances read: each encoder
// block's output of chunk, and each decoder step's prediction, recurrent state and joint logits, a streaming step one
// past its count since the reference's initialization forward took the first. It appends each step's chosen token to
// steps and counts the encoder boundaries in seen.
func (p transducerPath) observer(tensors *captureTensors, mode string, chunk int, steps *[]int, seen *int) *speechrecognition.TransducerObserver {
	return &speechrecognition.TransducerObserver{
		Encoder: func(trace speechrecognition.Trace) error {
			name := fmt.Sprintf("layer_%d", trace.Block)
			if trace.Block < 0 {
				name = "subsampling"
			}
			tensors.f32(fmt.Sprintf("%s/chunk_%d/%s", mode, chunk, name), trace.Values, 1, trace.Frames, trace.Width)
			*seen++
			return nil
		},
		Decoder: func(trace speechrecognition.DecoderTrace) error {
			step := trace.Step
			if mode == "streaming" {
				step++
			}
			key := fmt.Sprintf("%s/step_%d/", mode, step)
			hidden := len(trace.Hidden) / p.recurrent
			tensors.f32(key+"decoder", trace.Prediction, 1, 1, len(trace.Prediction))
			tensors.f32(key+"hidden", trace.Hidden, p.recurrent, 1, hidden)
			tensors.f32(key+"cell", trace.Cell, p.recurrent, 1, hidden)
			tensors.f32(key+"logits", trace.Logits, 1, 1, 1, len(trace.Logits))
			*steps = append(*steps, trace.Token)
			return nil
		},
	}
}
