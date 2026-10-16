package trainingdata

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestSpeechLatentMaterialization(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	id := func(kind artifact.Kind, text string) artifact.ID { return testutil.ArtifactID(t, kind, text) }
	profile := SpeechLatentProfile{Model: id(artifact.KindModel, "base"), Tokenizer: id(artifact.KindFile, "tokens"),
		Codec: id(artifact.KindTensorSet, "codec"), Transform: id(artifact.KindProfile, "normalization"),
		Conditioning: id(artifact.KindProfile, "text-bos"), Noise: id(artifact.KindProfile, "gaussian"),
		Width: 2, MaximumFrames: 3, MaximumTextBytes: 64, MaximumEncodedBytes: 2048}
	content, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	source := id(artifact.KindFile, "recorded input")
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "source", Artifacts: []artifact.Descriptor{{ID: source}}}); err != nil {
		t.Fatal(err)
	}
	process, err := SpeechLatentProcessor(store, profile)
	if err != nil {
		t.Fatal(err)
	}
	record := SpeechLatentRecord{Profile: content.Descriptor.ID, Source: source, SourceRecord: "row-12", Text: "a b", Frames: 2,
		Seed: -7, Latents: []float32{float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, 0.75, -0.125}}
	encode := func(record SpeechLatentRecord) RawRecord {
		t.Helper()
		data, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		return RawRecord{ID: "dataset-record-3", Group: "speaker-2", Data: data}
	}
	example, err := process(t.Context(), encode(record))
	if err != nil || example.ID != "dataset-record-3" || example.Group != "speaker-2" {
		t.Fatalf("source identity: %+v %v", example, err)
	}
	text, latents, frames, seed, err := SpeechLatents(example)
	if err != nil || text != record.Text || frames != record.Frames || seed != record.Seed || !slices.EqualFunc(latents, record.Latents, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
		t.Fatalf("pair=%q %v %d %d: %v", text, latents, frames, seed, err)
	}
	for _, failure := range []string{"profile", "source", "source-record", "shape", "frame-bound", "text-bound", "waveform", "speaker-conditioning", "encoded-bound", "nonfinite"} {
		t.Run(failure, func(t *testing.T) {
			bad := record
			switch failure {
			case "profile":
				bad.Profile = id(artifact.KindProfile, "other normalization")
			case "source":
				bad.Source = id(artifact.KindFile, "missing source")
			case "source-record":
				bad.SourceRecord = ""
			case "shape":
				bad.Latents = bad.Latents[:3]
			case "frame-bound":
				bad.Frames = 4
			case "text-bound":
				bad.Text = string(make([]byte, 65))
			}
			raw := encode(bad)
			switch failure {
			case "waveform", "speaker-conditioning":
				raw.Data = append(raw.Data[:len(raw.Data)-1], []byte(`,"`+failure+`":[0.1]}`)...)
			case "encoded-bound":
				raw.Data = make([]byte, profile.MaximumEncodedBytes+1)
			case "nonfinite":
				raw.Data = []byte(`{"normalized_codec_latents":[1e999]}`)
			}
			if got, err := process(t.Context(), raw); err == nil || len(got.Values) != 0 {
				t.Fatalf("accepted invalid pair: %+v %v", got, err)
			}
		})
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := process(ctx, encode(record)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled processor: %v", err)
	}
	profile.Transform = artifact.ID{}
	if _, err := SpeechLatentProcessor(store, profile); err == nil {
		t.Fatal("missing normalization admitted")
	}
}
