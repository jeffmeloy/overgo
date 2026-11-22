package audioparity

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

type encoderCapture struct {
	Version       int            `json:"version"`
	Source        SourceIdentity `json:"source"`
	Dataset       DatasetBinding `json:"dataset"`
	CaptureSHA256 string         `json:"capture_sha256"`
	TensorsSHA256 string         `json:"tensors_sha256"`
	TensorCount   int            `json:"tensor_count"`
	Runtime       struct {
		Python       string            `json:"python"`
		Torch        string            `json:"torch"`
		Torchaudio   string            `json:"torchaudio"`
		Transformers string            `json:"transformers"`
		Tokenizers   string            `json:"tokenizers"`
		Device       string            `json:"device"`
		Attention    string            `json:"attention"`
		Files        map[string]string `json:"files"`
	} `json:"runtime,omitzero"`
	// Producer names overgo as the capture's maker where overgo remade it (cmd/speech-capture); a capture the
	// reference software made names its runtime instead.
	Producer captureProducer `json:"producer,omitzero"`
	Cases    []encoderCase   `json:"cases"`
}

// encoderCase is one recording of an encoder capture: the corpus clip, its processed feature shape, the transcript the
// capture's maker decoded and the wall time of its forward.
type encoderCase struct {
	Fixture    string      `json:"fixture"`
	Audio      artifact.ID `json:"audio"`
	SampleRate uint32      `json:"sample_rate"`
	Samples    uint64      `json:"samples"`
	Frames     []int       `json:"frames"`
	Text       string      `json:"text"`
	WallNS     uint64      `json:"wall_ns"`
}

func (c encoderCapture) validate(e Election) error {
	if c.Version != 1 || c.Source != e.ModelSource || !reflect.DeepEqual(c.Dataset, e.Dataset) || len(c.Cases) == 0 || len(c.Cases) != len(e.Observations) || c.TensorCount <= 0 {
		return errors.New("audio capture: source, corpus, runtime or case count differs from election")
	}
	// The election's runtime is the reference software's: a capture overgo made is bound to the election's corpus and
	// transcripts, not to the runtime that elected them.
	reference := c.Runtime.Device == "cpu" && c.Runtime.Attention == "eager" && c.Runtime.Python == e.Runtime.Python && c.Runtime.Torch == e.Runtime.Torch &&
		c.Runtime.Transformers == e.Runtime.Transformers && len(c.Runtime.Files) != 0
	if err := captureMaker(c.Producer, c.Runtime, reference); err != nil {
		return fmt.Errorf("audio capture: source, corpus, runtime or case count differs from election: %w", err)
	}
	for i, x := range c.Cases {
		want := e.Observations[i]
		if x.Fixture != want.Fixture || x.Audio != want.Audio || x.SampleRate != want.SampleRate || x.Samples != want.Samples || x.Text != want.Observed || x.WallNS == 0 || len(x.Frames) != 3 || x.Frames[0] != 1 || x.Frames[1] <= 0 || x.Frames[2] <= 0 {
			return fmt.Errorf("audio capture: case %d differs from pinned observation", i)
		}
	}
	return nil
}

// PublishEncoderCapture registers native CPU reference tensors by their exact
// file identity and stores the source-bound capture document and capture script.
// It verifies identities and source bindings, not Go/reference numerical parity.
// Tensors remain an external read-only file; no corpus or model is copied.
func PublishEncoderCapture(ctx context.Context, repository artifact.Repository, election Election, reportPath, sourcePath, tensorsPath string) (artifact.ID, error) {
	if ctx == nil || repository == nil {
		return artifact.ID{}, errors.New("audio capture: incomplete execution")
	}
	if err := ctx.Err(); err != nil {
		return artifact.ID{}, err
	}
	data, err := artifact.ReadContentFile(reportPath)
	if err != nil {
		return artifact.ID{}, err
	}
	var capture encoderCapture
	if err = strictjson.DecodeBytes(data, &capture); err != nil {
		return artifact.ID{}, err
	}
	source, err := artifact.ReadContentFile(sourcePath)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishEncoderCapture(ctx, repository, election, capture, source, tensorsPath)
}

// publishEncoderCapture registers a validated encoder capture with the source that made it and its tensors file.
func publishEncoderCapture(ctx context.Context, repository artifact.Repository, election Election, capture encoderCapture, source []byte, tensorsPath string) (artifact.ID, error) {
	if err := capture.validate(election); err != nil {
		return artifact.ID{}, err
	}
	golden, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-encoder-golden/v1"), capture)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishTensorCapture(ctx, repository, golden, election.ID, capture.CaptureSHA256, source, capture.TensorsSHA256, capture.TensorCount, tensorsPath)
}

// RemakeEncoderCapture remakes the CPU ASR reference capture from overgo's own speech path: it runs each recording the
// election observed through the decoder, frontend, encoder, projection and CTC that TestCPUASRReference runs, keeps
// every boundary that acceptance compares in one tensors file under the references, registers the file where the store
// finds it, publishes the capture naming overgo at the remake's commit as its maker, and writes it to recordPath, the
// record the acceptance reads. It returns the published capture's identity.
func RemakeEncoderCapture(ctx context.Context, remake SpeechCaptureRemake, election Election, recordPath string) (artifact.ID, error) {
	if err := remake.ready(ctx); err != nil {
		return artifact.ID{}, err
	}
	directory, err := remake.directory("asr-reference")
	if err != nil {
		return artifact.ID{}, err
	}
	capture, tensors, err := traceEncoderCapture(ctx, remake.Store, election)
	if err != nil {
		return artifact.ID{}, err
	}
	path, err := tensors.write(directory)
	if err != nil {
		return artifact.ID{}, err
	}
	return recordEncoderCapture(ctx, remake.Store, election, capture, remake.producer(), path, recordPath)
}

// recordEncoderCapture names producer as the maker of capture, whose tensors lie at tensorsPath, publishes it and
// writes it to recordPath.
func recordEncoderCapture(ctx context.Context, repository artifact.Repository, election Election, capture encoderCapture, producer captureProducer, tensorsPath, recordPath string) (artifact.ID, error) {
	source, sourceSHA, tensorSHA, count, err := producer.recorded(tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	capture.Producer, capture.CaptureSHA256, capture.TensorsSHA256, capture.TensorCount = producer, sourceSHA, tensorSHA, count
	id, err := publishEncoderCapture(ctx, repository, election, capture, source, tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	return id, jsonfile.Write(recordPath, capture, 0o644)
}

// traceEncoderCapture runs every recording the election observed through overgo's speech path on the CPU, from the
// model and corpus the store records as the election's, and returns the capture's cases with every boundary
// TestCPUASRReference compares.
func traceEncoderCapture(ctx context.Context, store artifact.Reader, election Election) (encoderCapture, *captureTensors, error) {
	modelRoot, err := artifact.AvailablePath(ctx, store, election.Model.ID, artifact.LocationDirectory)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	for _, file := range election.Files {
		if file.Role == "weights" || file.Role == "tokenizer" || file.Role == "configuration" || file.Role == "preprocessor" {
			if err := verifyFile(modelRoot, file.Path, file.Artifact, file.Bytes); err != nil {
				return encoderCapture{}, nil, err
			}
		}
	}
	corpusPath, err := artifact.AvailablePath(ctx, store, election.Dataset.Artifact, artifact.LocationFile)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	if err := verifyFile(filepath.Dir(corpusPath), filepath.Base(corpusPath), election.Dataset.Artifact, election.Dataset.Bytes); err != nil {
		return encoderCapture{}, nil, err
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		return encoderCapture{}, nil, err
	}
	spec, known, err := RecognizerSpecFor(recipe.ModelType)
	if err != nil || !known {
		return encoderCapture{}, nil, errors.Join(errors.New("audio capture: the Granite recognizer is not reviewed"), err)
	}
	var declaration speechrecognition.Declaration
	if err := strictjson.DecodeBytes(spec.Execution, &declaration); err != nil {
		return encoderCapture{}, nil, err
	}
	weights, err := safetensors.OpenSource(modelRoot)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	defer weights.Close()
	encoder, err := speechrecognition.LoadEncoder(ctx, weights, declaration, captureMemoryBytes)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	defer encoder.Close(context.Background())
	tokenizer, err := hfbpe.Load(modelRoot)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	frontend, err := audiodsp.NewFrontend(spec.Frontend, captureMemoryBytes)
	if err != nil {
		return encoderCapture{}, nil, err
	}
	path := encoderPath{frontend: frontend, grouping: spec.Grouping, encoder: encoder, blocks: len(declaration.Blocks),
		vocabulary: int(recipe.Config.VocabSize), blank: spec.Blank, decode: tokenizer.Decode}
	capture := encoderCapture{Version: 1, Source: election.ModelSource, Dataset: election.Dataset}
	tensors := &captureTensors{}
	err = dataset.ReadParquetTextRows(ctx, corpusPath, "bytes", len(election.Observations), func(index uint64, payload string) error {
		if index >= uint64(len(election.Observations)) {
			return errors.New("audio capture: extra corpus row")
		}
		traced, err := path.trace(ctx, tensors, election.Observations[index], []byte(payload))
		capture.Cases = append(capture.Cases, traced)
		return err
	})
	if err != nil {
		return encoderCapture{}, nil, err
	}
	if len(capture.Cases) != len(election.Observations) {
		return encoderCapture{}, nil, errors.New("audio capture: the corpus holds fewer recordings than the election observed")
	}
	return capture, tensors, nil
}

// encoderPath is overgo's CPU speech path for one connectionist checkpoint: its frontend and grouping, its encoder of
// blocks blocks, the vocabulary its projection reads and its blank entry, and its tokenizer's decoding.
type encoderPath struct {
	frontend   *audiodsp.Frontend
	grouping   audiodsp.GroupedFeatureConfig
	encoder    *speechrecognition.Encoder
	blocks     int
	vocabulary int
	blank      int
	decode     func([]int) string
}

// trace runs one encoded recording through the path, keeping under its fixture's name the boundaries
// TestCPUASRReference compares: the decoded waveform, the grouped features, their all-valid mask, the input projection
// and each block's output, the logits and the CTC frame identifiers.
func (p encoderPath) trace(ctx context.Context, tensors *captureTensors, want TranscriptObservation, payload []byte) (encoderCase, error) {
	id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
	if err != nil || id != want.Audio {
		return encoderCase{}, errors.Join(fmt.Errorf("audio capture: %s: encoded audio identity differs", want.Fixture), err)
	}
	audio, _, err := media.DecodeAudio(ctx, payload, want.Samples)
	if err != nil {
		return encoderCase{}, err
	}
	if uint64(len(audio.Samples)) != want.Samples || audio.Format.SampleRate != uint64(want.SampleRate) || audio.Format.Channels != 1 {
		return encoderCase{}, fmt.Errorf("audio capture: %s: decoded audio geometry differs", want.Fixture)
	}
	prefix := want.Fixture + "/"
	start := time.Now()
	var fw audiodsp.Workspace
	var ew speechrecognition.Workspace
	features, frames, width, err := p.frontend.ProcessGrouped(ctx, audio.Samples, int(want.SampleRate), &fw, p.grouping)
	if err != nil {
		return encoderCase{}, err
	}
	valid := make([]int, frames)
	for index := range valid {
		valid[index] = 1
	}
	tensors.f32(prefix+"waveform", audio.Samples, len(audio.Samples))
	tensors.f32(prefix+"features", features, frames, width)
	tensors.ids(prefix+"mask", valid, frames)
	seen := 0
	hidden, outFrames, err := p.encoder.Encode(ctx, features, frames, &ew, func(trace speechrecognition.Trace) error {
		if trace.Block != seen-1 {
			return errors.New("audio capture: missing or reordered encoder boundary")
		}
		name := fmt.Sprintf("layer_%d", trace.Block)
		if trace.Block < 0 {
			name = "input_projection"
		}
		tensors.f32(prefix+name, trace.Values, trace.Frames, trace.Width)
		seen++
		return nil
	})
	if err != nil {
		return encoderCase{}, err
	}
	if seen != p.blocks+1 || outFrames <= 0 {
		return encoderCase{}, fmt.Errorf("audio capture: %s: incomplete encoder boundaries", want.Fixture)
	}
	logits, err := p.encoder.Project(ctx, hidden, outFrames, &ew)
	if err != nil {
		return encoderCase{}, err
	}
	tensors.f32(prefix+"logits", logits, outFrames, p.vocabulary)
	frameIDs := make([]int, outFrames)
	ids, err := speechrecognition.GreedyCTC(ctx, frameIDs, make([]int, outFrames), logits, outFrames, p.vocabulary, p.blank)
	if err != nil {
		return encoderCase{}, err
	}
	tensors.ids(prefix+"tokens", frameIDs, outFrames)
	return encoderCase{Fixture: want.Fixture, Audio: want.Audio, SampleRate: want.SampleRate, Samples: want.Samples,
		Frames: []int{1, frames, width}, Text: p.decode(ids), WallNS: uint64(time.Since(start))}, nil
}

func publishTensorCapture(ctx context.Context, repository artifact.Repository, golden artifact.Content, parent artifact.ID, sourceSHA string, source []byte, tensorSHA string, tensorCount int, tensorsPath string) (artifact.ID, error) {
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		return artifact.ID{}, err
	}
	if sourceID.DigestHex() != sourceSHA {
		return artifact.ID{}, errors.New("audio capture: capture script hash differs")
	}
	tensorID, size, count, err := captureTensorFile(tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	if tensorID.DigestHex() != tensorSHA {
		return artifact.ID{}, errors.New("audio capture: tensor file hash differs")
	}
	if count != tensorCount {
		return artifact.ID{}, errors.New("audio capture: tensor count differs")
	}
	script, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-oracle-capture-source/v1"), struct{ SHA256, Source string }{sourceSHA, string(source)})
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch("audio/encoder-golden/"+golden.Descriptor.ID.DigestHex(), []artifact.Content{script, golden}, artifact.DependencyLineage(golden.Descriptor.ID, parent, script.Descriptor.ID, tensorID), nil)
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: tensorID, Size: size, MediaType: "application/vnd.safetensors"})
	location, err := artifact.CanonicalLocalLocation(tensorID, artifact.LocationFile, tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	locationID, err := artifact.JSONID(artifact.KindEvidence, location)
	if err != nil {
		return artifact.ID{}, err
	}
	// Availability belongs in the replay identity: the same capture can be
	// installed at another location without reusing a key for different bytes.
	batch.Key += "/" + locationID.DigestHex()
	batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	if _, err = artifact.Publish(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return golden.Descriptor.ID, nil
}
