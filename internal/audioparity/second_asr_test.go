package audioparity

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

type transducerDeclaration struct {
	Frontend   audiodsp.FrontendConfig             `json:"frontend"`
	Encoder    speechrecognition.Declaration       `json:"encoder"`
	Transducer speechrecognition.TransducerBinding `json:"transducer"`
}

type transducerFixture struct {
	declaration transducerDeclaration
	model       *speechrecognition.Transducer
	frontend    *audiodsp.Frontend
	tokenizer   *hfbpe.Tokenizer
	store       *overgodb.Store
	captures    []artifact.ID
	modelID     artifact.ID
}

func loadTransducerFixture(t *testing.T) *transducerFixture {
	t.Helper()
	f := &transducerFixture{}
	data, err := os.ReadFile("recipes/transducer_execution.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := strictjson.DecodeBytes(data, &f.declaration); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile("testdata/transducer_captures.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := strictjson.DecodeBytes(data, &f.captures); err != nil || len(f.captures) != 2 {
		t.Fatalf("capture denominator: %v", err)
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	f.store, err = overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.store.Close() })
	first := f.capture(t, f.captures[0])
	f.modelID = first.Model
	registration, err := os.ReadFile("testdata/registered_models.json")
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Model    artifact.ID `json:"model"`
		Revision string      `json:"source_commit"`
		License  struct {
			SPDX string `json:"spdx"`
		} `json:"license"`
	}
	if err := json.Unmarshal(registration, &entries); err != nil {
		t.Fatal(err)
	}
	bound := false
	for _, entry := range entries {
		bound = bound || entry.Model == first.Model && entry.Revision == first.Revision && entry.License.SPDX == first.License
	}
	if !bound {
		t.Fatal("capture model/source/license differs from registered declaration")
	}
	directory, err := artifact.AvailablePath(t.Context(), f.store, first.Model, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	for name, digest := range first.Files {
		file, err := os.Open(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		id, _, err := artifact.Identify(artifact.KindFile, file)
		file.Close()
		if err != nil || id.DigestHex() != digest {
			t.Fatalf("model member %s changed: %v", name, err)
		}
	}
	weights, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	// Test allocation ceiling only; neither model peak RSS nor a performance target.
	f.model, err = speechrecognition.LoadTransducer(t.Context(), weights, f.declaration.Encoder, f.declaration.Transducer, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	f.frontend, err = audiodsp.NewFrontend(f.declaration.Frontend, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	f.tokenizer, err = hfbpe.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *transducerFixture) capture(t *testing.T, id artifact.ID) transducerCapture {
	t.Helper()
	content, err := artifact.RequireTypedContent(t.Context(), f.store, id)
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.Schema != transducerCaptureContract.Schema {
		t.Fatal("wrong capture schema")
	}
	var capture transducerCapture
	if err := strictjson.DecodeBytes(content.Data, &capture); err != nil {
		t.Fatal(err)
	}
	if err := capture.validate(); err != nil {
		t.Fatal(err)
	}
	if f.modelID.Valid() && capture.Model != f.modelID {
		t.Fatal("capture changed model within the fixture group")
	}
	return capture
}

func (f *transducerFixture) inputs(t *testing.T, capture transducerCapture) ([]float32, int, *safetensors.Source) {
	t.Helper()
	tensorID, err := artifact.ParseID("file:sha256:" + capture.TensorsSHA256)
	if err != nil {
		t.Fatal(err)
	}
	path, err := artifact.AvailablePath(t.Context(), f.store, tensorID, artifact.LocationFile)
	if err != nil {
		t.Fatal(err)
	}
	oracle, err := safetensors.OpenSource(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { oracle.Close() })
	digests, err := oracle.ShardDigests()
	if err != nil || len(digests) != 1 || hex.EncodeToString(digests[0].Digest[:]) != capture.TensorsSHA256 || len(oracle.Tensors) != capture.TensorCount {
		t.Fatalf("oracle bytes or tensor count differ: %v", err)
	}
	shardID, err := artifact.ParseID("file:sha256:" + capture.Dataset.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	shard := librispeechPath(t, filepath.FromSlash(capture.Dataset.Path))
	file, err := os.Open(shard)
	if err != nil {
		t.Fatal(err)
	}
	actual, _, err := artifact.Identify(artifact.KindFile, file)
	file.Close()
	if err != nil || actual != shardID {
		t.Fatalf("source shard changed: %v", err)
	}
	var pcm []float32
	err = dataset.ReadParquetTextRows(t.Context(), shard, "bytes", capture.Dataset.Row+1, func(row uint64, payload string) error {
		if row != uint64(capture.Dataset.Row) {
			return nil
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, []byte(payload))
		if err != nil || id.DigestHex() != capture.AudioSHA256 {
			return errors.Join(errors.New("source audio changed"), err)
		}
		audio, _, err := media.DecodeAudio(t.Context(), []byte(payload), uint64(capture.Samples))
		if err != nil {
			return err
		}
		if audio.Format.SampleRate != uint64(capture.SampleRate) || audio.Format.Channels != 1 {
			return errors.New("source audio format differs")
		}
		pcm = audio.Samples
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(pcm, captureF32(t, oracle, "waveform", capture.Samples)) {
		t.Fatal("native decoded PCM differs")
	}
	var workspace audiodsp.Workspace
	features, frames, err := f.frontend.Process(t.Context(), [][]float32{pcm}, capture.SampleRate, &workspace, audiodsp.ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// The pinned processor zeros the centered STFT's final invalid frame.
	valid := len(pcm) / int(f.declaration.Frontend.Geometry.HopSamples)
	mask, ok := oracle.Tensors["mask"]
	if !ok || mask.DType != "BOOL" || !slices.Equal(mask.Shape, []uint64{uint64(frames)}) {
		t.Fatal("invalid frontend mask trace")
	}
	maskBytes, err := io.ReadAll(mask.Reader())
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range maskBytes {
		if (b != 0) != (i < valid) {
			t.Fatal("frontend valid-frame policy differs")
		}
	}
	assertASRBoundary(t, "transducer frontend", features, captureF32(t, oracle, "features", frames, f.declaration.Transducer.Bands), f.declaration.Frontend.FFTLength)
	stream, err := audiodsp.NewStreamFrontend(f.declaration.Frontend, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	var state audiodsp.StreamState
	var streamed []float32
	// Non-hop-aligned chunks exercise waveform continuity independently of the
	// model's mel-chunk schedule. Restart every boundary with no hidden scratch.
	chunkSamples := capture.SampleRate + 1
	for start := 0; start < len(pcm); start += chunkSamples {
		end := min(start+chunkSamples, len(pcm))
		var live audiodsp.StreamWorkspace
		chunk, _, next, err := stream.Process(t.Context(), pcm[start:end], capture.SampleRate, state, end == len(pcm), &live)
		if err != nil {
			t.Fatal(err)
		}
		streamed, state = append(streamed, chunk...), next
	}
	if !slices.Equal(streamed, features) || state.Frames != uint64(frames) {
		t.Fatal("waveform streaming differs from native-bound offline features")
	}
	return features, frames, oracle
}

func transducerObserver(t *testing.T, f *transducerFixture, oracle *safetensors.Source, mode string, chunk int, seen *int) *speechrecognition.TransducerObserver {
	t.Helper()
	return &speechrecognition.TransducerObserver{
		Encoder: func(trace speechrecognition.Trace) error {
			name := fmt.Sprintf("layer_%d", trace.Block)
			if trace.Block < 0 {
				name = "subsampling"
			}
			key := fmt.Sprintf("%s/chunk_%d/%s", mode, chunk, name)
			assertASRBoundary(t, key, trace.Values, captureF32(t, oracle, key, 1, trace.Frames, trace.Width), trace.Width)
			*seen++
			return nil
		},
		Decoder: func(trace speechrecognition.DecoderTrace) error {
			step := trace.Step
			// Native streaming generate performs one initialization forward whose
			// joint output is unused, before creating its generation decoder cache.
			if mode == "streaming" {
				step++
			}
			key := fmt.Sprintf("%s/step_%d/", mode, step)
			d := f.declaration.Transducer
			hidden := len(trace.Hidden) / len(d.Recurrent)
			assertASRBoundary(t, key+"decoder", trace.Prediction, captureF32(t, oracle, key+"decoder", 1, 1, len(trace.Prediction)), hidden)
			assertASRBoundary(t, key+"hidden", trace.Hidden, captureF32(t, oracle, key+"hidden", len(d.Recurrent), 1, hidden), hidden)
			assertASRBoundary(t, key+"cell", trace.Cell, captureF32(t, oracle, key+"cell", len(d.Recurrent), 1, hidden), hidden)
			assertASRBoundary(t, key+"logits", trace.Logits, captureF32(t, oracle, key+"logits", 1, 1, 1, len(trace.Logits)), len(trace.Prediction))
			return nil
		},
	}
}

func TestSecondASROfflineAcceptance(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": exact second-ASR acceptance requires registered native traces")
	}
	f := loadTransducerFixture(t)
	for _, id := range f.captures {
		capture := f.capture(t, id)
		t.Run(capture.Fixture, func(t *testing.T) {
			x, frames, oracle := f.inputs(t, capture)
			var w speechrecognition.TransducerWorkspace
			seen := 0
			result, err := f.model.Recognize(t.Context(), x, frames, &w, transducerObserver(t, f, oracle, "offline", 0, &seen))
			if err != nil {
				t.Fatal(err)
			}
			if seen != len(f.declaration.Encoder.Blocks)+1 {
				t.Fatal("encoder boundary count differs")
			}
			requireTransducerResult(t, f, oracle, capture, "offline", result.Tokens, result.Durations)
			tokens, durations := slices.Clone(result.Tokens), slices.Clone(result.Durations)
			again, err := f.model.Recognize(t.Context(), x, frames, &w, nil)
			if err != nil || !slices.Equal(tokens, again.Tokens) || !slices.Equal(durations, again.Durations) {
				t.Fatalf("unobserved replay differs: %v", err)
			}
			t.Logf("%d source samples; %d encoder boundaries; %d exact emissions and frame advances; all recurrent states and joint logits compared", capture.Samples, seen, len(tokens))
		})
	}
	t.Run("first-architecture-preserved", TestCPUASRReference)
	t.Run("native-artifact-registration", TestCanonicalAudioModelRegistration)
}

func requireTransducerResult(t *testing.T, f *transducerFixture, oracle *safetensors.Source, capture transducerCapture, mode string, tokens, durations []int) {
	t.Helper()
	for _, pair := range []struct {
		name string
		got  []int
	}{{"sequences", tokens}, {"durations", durations}} {
		tensor := oracle.Tensors[mode+"/"+pair.name]
		if len(tensor.Shape) != 2 || tensor.Shape[0] != 1 {
			t.Fatal("invalid emission geometry")
		}
		want := captureIDs(t, oracle, mode+"/"+pair.name, 1, int(tensor.Shape[1]))
		if !slices.Equal(pair.got, want) {
			t.Fatalf("%s %s differs", mode, pair.name)
		}
	}
	text, err := f.tokenizer.DecodeText(tokens)
	if err != nil || text != capture.Captures[mode].Text {
		t.Fatalf("%s text=%q want=%q: %v", mode, text, capture.Captures[mode].Text, err)
	}
}

func TestSecondASRStreamingAcceptance(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": exact second-ASR acceptance requires cache-aware streaming traces")
	}
	f := loadTransducerFixture(t)
	for _, id := range f.captures {
		capture := f.capture(t, id)
		t.Run(capture.Fixture, func(t *testing.T) {
			x, frames, oracle := f.inputs(t, capture)
			var w speechrecognition.TransducerWorkspace
			var tokens, durations []int
			seen, chunks := 0, 0
			for offset, size := 0, capture.FirstChunkFrames; offset < frames; offset, size = offset+size, capture.FollowingChunkFrames {
				chunk := make([]float32, size*f.declaration.Transducer.Bands)
				copy(chunk, x[offset*f.declaration.Transducer.Bands:min(frames, offset+size)*f.declaration.Transducer.Bands])
				result, err := f.model.RecognizeChunk(t.Context(), chunk, size, offset+size >= frames, &w, transducerObserver(t, f, oracle, "streaming", chunks, &seen))
				if err != nil {
					t.Fatal(err)
				}
				tokens = append(tokens, result.Tokens...)
				durations = append(durations, result.Durations...)
				chunks++
			}
			if chunks != len(capture.Captures["streaming"].Chunks) || seen != chunks*(len(f.declaration.Encoder.Blocks)+1) {
				t.Fatal("stream boundary denominator differs")
			}
			requireTransducerResult(t, f, oracle, capture, "streaming", tokens, durations)
			if _, err := f.model.RecognizeChunk(t.Context(), make([]float32, capture.FollowingChunkFrames*f.declaration.Transducer.Bands), capture.FollowingChunkFrames, true, &w, nil); err == nil {
				t.Fatal("reused finalized stream")
			}
			var cancelled speechrecognition.TransducerWorkspace
			ctx, cancel := context.WithCancelCause(t.Context())
			_, err := f.model.RecognizeChunk(ctx, x[:capture.FirstChunkFrames*f.declaration.Transducer.Bands], capture.FirstChunkFrames, false, &cancelled, &speechrecognition.TransducerObserver{Encoder: func(speechrecognition.Trace) error { cancel(nil); return ctx.Err() }})
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("encoder cancellation=%v", err)
			}
			if _, err := f.model.RecognizeChunk(t.Context(), x[:capture.FollowingChunkFrames*f.declaration.Transducer.Bands], capture.FollowingChunkFrames, false, &cancelled, nil); err == nil {
				t.Fatal("resumed partially mutated failed cache")
			}
			t.Logf("%d source samples; %d chunks; %d encoder boundaries; %d exact emissions/advances; cache rollover, finalization and cancellation checked", capture.Samples, chunks, seen, len(tokens))
		})
	}
}
