package speechrecognitiontest

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// NativeClip binds decoded corpus samples to a content-addressed native capture.
// Offline and Streaming are deliberately distinct expected transcripts.
type NativeClip struct {
	Capture            artifact.ID
	Name               string
	PCM                []float32
	Offline, Streaming string
}

// NativeFixture supplies one operation-declared recurrent recipe and the pinned
// native corpus captures. It never invokes a Python runtime or copies weights.
type NativeFixture struct {
	Definition recipe.Definition
	Profile    speechrecognition.ExecutionProfile
	Clips      []NativeClip
}

// PublishNative registers the recurrent fixture for session and HTTP consumers.
// The numerical boundary acceptance remains owned by internal/audioparity.
func PublishNative(t *testing.T, destination artifact.Repository) NativeFixture {
	t.Helper()
	repo := testutil.RepoRoot(t)
	read := func(name string, target any) {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(repo, "internal", "audioparity", filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		if err := strictjson.DecodeBytes(data, target); err != nil {
			t.Fatal(err)
		}
	}
	var declaration struct {
		Frontend   audiodsp.FrontendConfig             `json:"frontend"`
		Encoder    speechrecognition.Declaration       `json:"encoder"`
		Transducer speechrecognition.TransducerBinding `json:"transducer"`
	}
	read("recipes/transducer_execution.json", &declaration)
	var captures []artifact.ID
	read("testdata/transducer_captures.json", &captures)
	if len(captures) == 0 {
		t.Fatal("native stream fixture has no registered captures")
	}
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	source, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	profile, err := speechrecognition.NewExecutionProfile(speechrecognition.ExecutionProfile{
		Frontend: declaration.Frontend, Grouping: audiodsp.GroupedFeatureConfig{StackFrames: 1},
		Encoder: declaration.Encoder, Transducer: &declaration.Transducer, BlankToken: declaration.Transducer.Blank, Language: "en",
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := NativeFixture{Profile: profile}
	for _, id := range captures {
		content, err := artifact.RequireTypedContent(t.Context(), source, id)
		if err != nil || content.Descriptor.Schema != "overgo/audio-transducer-golden/v1" {
			t.Fatalf("native capture: %v", err)
		}
		// A typed projection of the immutable capture, not a second capture schema.
		var capture struct {
			Model         artifact.ID `json:"model"`
			Fixture       string      `json:"fixture"`
			Samples       int         `json:"samples"`
			SampleRate    int         `json:"sample_rate"`
			TensorsSHA256 string      `json:"tensors_sha256"`
			Captures      map[string]struct {
				Text string `json:"text"`
			} `json:"captures"`
		}
		if err := json.Unmarshal(content.Data, &capture); err != nil || capture.Samples <= 0 || capture.SampleRate != profile.Frontend.SampleRate {
			t.Fatalf("capture input geometry: %v", err)
		}
		if !fixture.Definition.ID.Valid() {
			directory, err := artifact.AvailablePath(t.Context(), source, capture.Model, artifact.LocationDirectory)
			if err != nil {
				t.Fatal(err)
			}
			fixture.Definition = PublishModel(t, destination, directory, profile)
		}
		if fixture.Definition.Model != capture.Model {
			t.Fatal("registered model bytes differ from native capture")
		}
		tensorID, err := artifact.ParseID("file:sha256:" + capture.TensorsSHA256)
		if err != nil {
			t.Fatal(err)
		}
		path, err := artifact.AvailablePath(t.Context(), source, tensorID, artifact.LocationFile)
		if err != nil {
			t.Fatal(err)
		}
		oracle, err := safetensors.OpenSource(filepath.Dir(path))
		if err != nil {
			t.Fatal(err)
		}
		digests, err := oracle.ShardDigests()
		if err != nil || len(digests) != 1 || hex.EncodeToString(digests[0].Digest[:]) != capture.TensorsSHA256 {
			oracle.Close()
			t.Fatalf("capture tensor bytes differ: %v", err)
		}
		wave, found := oracle.Tensors["waveform"]
		if !found || wave.DType != "F32" || !slices.Equal(wave.Shape, []uint64{uint64(capture.Samples)}) {
			oracle.Close()
			t.Fatal("captured waveform is absent or differs")
		}
		pcm, err := safetensors.ReadF32(wave)
		oracle.Close()
		if err != nil {
			t.Fatal(err)
		}
		fixture.Clips = append(fixture.Clips, NativeClip{Capture: id, Name: capture.Fixture, PCM: pcm,
			Offline: capture.Captures["offline"].Text, Streaming: capture.Captures["streaming"].Text})
	}
	return fixture
}

// NativeWave encodes corpus PCM without changing a sample. The round-trip
// assertion refuses using this helper for samples that PCM16 cannot represent.
func NativeWave(t *testing.T, pcm []float32, rate int) []byte {
	t.Helper()
	values := make([]int16, len(pcm))
	for i, x := range pcm {
		values[i] = int16(x * (1 << 15))
	}
	data := testutil.MonoPCM16WAV(uint32(rate), values)
	audio, _, err := media.DecodeAudio(t.Context(), data, uint64(len(pcm)))
	if err != nil || !slices.Equal(audio.Samples, pcm) {
		t.Fatalf("native waveform transport changed samples: %v", err)
	}
	return data
}

// PublishStream registers sample-exact waveform chunks and their bounded policy.
func (fixture NativeFixture) PublishStream(t *testing.T, store artifact.Repository, clip NativeClip, chunkSamples int) (recipecontract.AudioReference, []workflowruntime.AudioStreamChunk) {
	t.Helper()
	if chunkSamples <= 0 {
		t.Fatal("native stream: invalid test chunk size")
	}
	commit := func(batch artifact.Batch, err error) {
		t.Helper()
		if err == nil {
			_, err = artifact.CommitBatch(t.Context(), store, batch)
		}
		if err != nil && !errors.Is(err, artifact.ErrNoChange) {
			t.Fatal(err)
		}
	}
	publish := func(pcm []float32) artifact.ID {
		t.Helper()
		data := NativeWave(t, pcm, fixture.Profile.Frontend.SampleRate)
		id, err := artifact.IdentifyBytes(artifact.KindFile, data)
		if err != nil {
			t.Fatal(err)
		}
		commit(artifact.Batch{Key: "native-stream/wave/" + id.String(), Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data}}}, nil)
		return id
	}
	source := recipecontract.AudioReference{Audio: publish(clip.PCM)}
	policy := workflowruntime.AudioStreamPolicy{Model: fixture.Definition.Model,
		Format:          recipecontract.AudioFormat{SampleRate: uint64(fixture.Profile.Frontend.SampleRate), Channels: 1, Encoding: "pcm-f32le"},
		MaxChunkSamples: uint64(chunkSamples)}
	batch, err := policy.Batch("native-stream/policy")
	commit(batch, err)
	source.Profile = batch.Contents[0].Descriptor.ID
	var chunks []workflowruntime.AudioStreamChunk
	for start := 0; start < len(clip.PCM); start += chunkSamples {
		end := min(start+chunkSamples, len(clip.PCM))
		chunks = append(chunks, workflowruntime.AudioStreamChunk{Sequence: uint64(len(chunks)), Audio: publish(clip.PCM[start:end]),
			Span: recipecontract.SampleSpan{Start: uint64(start), End: uint64(end)}, Final: end == len(clip.PCM)})
	}
	return source, chunks
}
