package speechrecognition

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
)

const transcriptionTestCommit = "0123456789abcdef0123456789abcdef01234567"

func TestTranscriptionRecipeLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	modelDirectory, declaration := transcriptionFixtureModel(t)
	hf, err := hfrepo.Open(modelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromHFRepository(hf)
	if closeErr := hf.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	inventoryBatch, err := inventory.Batch("fixture/transcription/model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), store, inventoryBatch); err != nil {
		t.Fatal(err)
	}
	tokenizerID := tokenizerComponent(t, inventory.Manifest)

	frontend := audiodsp.FrontendConfig{
		SampleRate: 16_000,
		Geometry:   recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 2, FeatureBins: 4},
		FFTLength:  16, FrameSpan: 4, Padding: "zero", Window: "rectangular",
		Mel: audiodsp.MelConfig{Scale: "htk", MinFrequency: 20, MaxFrequency: 7_000},
		Log: audiodsp.LogConfig{Power: true, Base: "natural", GuardMode: "clamp", Guard: 1e-8, Scale: 1},
	}
	grouping := audiodsp.GroupedFeatureConfig{StackFrames: 1, FinalFrameSamples: 4}
	profile, err := NewExecutionProfile(frontend, grouping, declaration, 0, "en")
	if err != nil {
		t.Fatal(err)
	}
	contract, err := modelrecipe.NewAudioContract(
		recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		frontend.Geometry, artifact.ID{}, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	profileBatch, err := profile.Batch("fixture/transcription/profile")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), store, profileBatch); err != nil {
		t.Fatal(err)
	}
	contractBatch, err := contract.Batch("fixture/transcription/contract")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), store, contractBatch); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.TranscriptionDefinition(
		inventory.Manifest.ID, contract.ID, profile.ID, tokenizerID, inventory.TensorInventory.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = modelrecipe.PublishCandidate(t.Context(), store, "fixture/transcription/candidate", definition); err != nil {
		t.Fatal(err)
	}

	environment := transcriptionEvidence(t, "cpu-fixture")
	wave := transcriptionWAV(testWave(128))
	silence := transcriptionWAV(make([]float32, 128))
	registerTranscriptionInputs(t, store, environment, wave, silence)
	session, err := LoadSession(t.Context(), store, definition.ID, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))
	transcriber, err := session.Lease(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer transcriber.Release()
	policy := dataset.AudioInspectionPolicy{
		MaximumEncodedBytes: 1 << 20, MaximumSamples: 1 << 16, ClipThreshold: .999,
		Admission: recipecontract.AudioAdmissionPolicy{
			MinimumDurationNanoseconds: uint64(time.Millisecond), MaximumDurationNanoseconds: uint64(time.Second),
			MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: .001,
			MaximumClippedFraction: 0, MaximumAbsoluteDCOffset: .1,
		},
	}
	binding := RunBinding{Key: "fixture/transcription/run/accepted", CodeCommit: transcriptionTestCommit, Environment: environment.Descriptor.ID}
	waveID, _ := artifact.IdentifyBytes(artifact.KindFile, wave)
	result, run, err := transcriber.Transcribe(t.Context(), wave, dataset.AudioPayloadOrigin{Container: waveID}, policy, binding)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text == "" || result.Language != "en" || result.Source.Audio != waveID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Outputs) != 1 {
		t.Fatalf("transcription = %+v, run = %+v", result, run)
	}
	if !slices.Contains(run.Inputs, inventory.Manifest.ID) {
		t.Fatalf("transcription run does not bind model %s: %+v", inventory.Manifest.ID, run.Inputs)
	}
	loadedRun, err := runrecord.RequireExactRun(t.Context(), store, run.ID)
	if err != nil || loadedRun.ID != run.ID {
		t.Fatalf("exact run = %+v, %v", loadedRun, err)
	}
	output, err := artifact.RequireTypedContent(t.Context(), store, run.Outputs[0])
	if err != nil || output.Descriptor.Schema != TranscriptionSchema {
		t.Fatalf("transcription output = %+v, %v", output.Descriptor, err)
	}
	var persisted recipecontract.Transcription
	if err = json.Unmarshal(output.Data, &persisted); err != nil || persisted != result {
		t.Fatalf("persisted transcription = %+v, %v", persisted, err)
	}

	second, secondRun, err := transcriber.Transcribe(t.Context(), wave, dataset.AudioPayloadOrigin{Container: waveID}, policy,
		RunBinding{Key: "fixture/transcription/run/replay", CodeCommit: transcriptionTestCommit, Environment: environment.Descriptor.ID})
	if err != nil || second != result || !slices.Equal(secondRun.Outputs, run.Outputs) {
		t.Fatalf("deterministic replay = %+v, %+v, %v", second, secondRun, err)
	}

	silenceID, _ := artifact.IdentifyBytes(artifact.KindFile, silence)
	_, refusedRun, err := transcriber.Transcribe(t.Context(), silence, dataset.AudioPayloadOrigin{Container: silenceID}, policy,
		RunBinding{Key: "fixture/transcription/run/refused", CodeCommit: transcriptionTestCommit, Environment: environment.Descriptor.ID})
	if !errors.Is(err, ErrAudioAdmissionRefused) || refusedRun.Outcome != runrecord.OutcomeFailed || len(refusedRun.Outputs) != 0 {
		t.Fatalf("refused run = %+v, %v", refusedRun, err)
	}
	if _, err = runrecord.RequireExactRun(t.Context(), store, refusedRun.ID); err != nil {
		t.Fatal(err)
	}

	_, beforeCancel := store.Head()
	cancelled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	_, cancelledRun, err := transcriber.Transcribe(cancelled, wave, dataset.AudioPayloadOrigin{Container: waveID}, policy,
		RunBinding{Key: "fixture/transcription/run/cancelled", CodeCommit: transcriptionTestCommit, Environment: environment.Descriptor.ID})
	_, afterCancel := store.Head()
	if !errors.Is(err, context.Canceled) || cancelledRun.ID.Valid() || afterCancel != beforeCancel {
		t.Fatalf("cancelled run = %+v, %v; commits %d -> %d", cancelledRun, err, beforeCancel, afterCancel)
	}
	t.Run("shared-component-lifetime", func(t *testing.T) {
		session, err := LoadSession(t.Context(), store, definition.ID, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		defer session.Close(context.WithoutCancel(t.Context()))
		lease, err := session.Lease(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		defer lease.Release()
		if _, err := session.Lease(cancelled); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled admission: %v", err)
		}
		got, _, err := lease.Transcribe(t.Context(), wave, dataset.AudioPayloadOrigin{Container: waveID}, policy,
			RunBinding{Key: "fixture/shared-session", CodeCommit: transcriptionTestCommit, Environment: environment.Descriptor.ID})
		if err != nil || got != result {
			t.Fatalf("shared session changed clip execution: %+v %v", got, err)
		}
		if err := lease.Release(); err != nil {
			t.Fatal(err)
		}
		if _, _, err := lease.Transcribe(t.Context(), wave, dataset.AudioPayloadOrigin{Container: waveID}, policy, binding); err == nil {
			t.Fatal("released lease executed")
		}
		if snapshot := session.Snapshot(); snapshot.Active != 0 || snapshot.Waiting != 0 || snapshot.Loads != 1 {
			t.Fatalf("leaked component: %+v", snapshot)
		}
		if err := session.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, err := session.Lease(t.Context()); err == nil {
			t.Fatal("closed session admitted")
		}
	})
}

func transcriptionFixtureModel(t *testing.T) (string, Declaration) {
	t.Helper()
	encoded, err := os.ReadFile("testdata/encoder.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture smallFixture
	if err = json.Unmarshal(encoded, &fixture); err != nil {
		t.Fatal(err)
	}
	values := make(map[string][]float32, len(fixture.Weights))
	shapes := make(map[string][]int, len(fixture.Weights))
	for name, tensor := range fixture.Weights {
		values[name], shapes[name] = tensor.Values, tensor.Shape
	}
	directory := t.TempDir()
	if err = safetensors.Save(filepath.Join(directory, "model.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(directory, "config.json"), []byte(`{"model_type":"fixture"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer := []byte(`{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"x":3,"y":4},"merges":[],"byte_fallback":false}}`)
	if err = os.WriteFile(filepath.Join(directory, "tokenizer.json"), tokenizer, 0o600); err != nil {
		t.Fatal(err)
	}
	return directory, fixture.Declaration
}

func tokenizerComponent(t *testing.T, manifest artifact.Manifest) artifact.ID {
	t.Helper()
	for _, component := range manifest.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == "tokenizer.json" {
			return component.Artifact
		}
	}
	t.Fatal("tokenizer.json component is absent")
	return artifact.ID{}
}

func transcriptionEvidence(t *testing.T, name string) artifact.Content {
	t.Helper()
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/test-transcription-environment/v1"), map[string]string{"name": name})
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func registerTranscriptionInputs(t *testing.T, store artifact.Repository, environment artifact.Content, payloads ...[]byte) {
	t.Helper()
	batch := artifact.Batch{Key: "fixture/transcription/inputs", Contents: []artifact.Content{environment}}
	for _, payload := range payloads {
		id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
		if err != nil {
			t.Fatal(err)
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id, Size: uint64(len(payload))})
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
}

func testWave(samples int) []float32 {
	values := make([]float32, samples)
	for index := range values {
		values[index] = .25 * float32(math.Sin(2*math.Pi*float64(index)/16))
	}
	return values
}

func transcriptionWAV(samples []float32) []byte {
	var buffer bytes.Buffer
	dataSize := uint32(len(samples) * 2)
	buffer.WriteString("RIFF")
	_ = binary.Write(&buffer, binary.LittleEndian, uint32(36)+dataSize)
	buffer.WriteString("WAVEfmt ")
	_ = binary.Write(&buffer, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buffer, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buffer, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buffer, binary.LittleEndian, uint32(16_000))
	_ = binary.Write(&buffer, binary.LittleEndian, uint32(32_000))
	_ = binary.Write(&buffer, binary.LittleEndian, uint16(2))
	_ = binary.Write(&buffer, binary.LittleEndian, uint16(16))
	buffer.WriteString("data")
	_ = binary.Write(&buffer, binary.LittleEndian, dataSize)
	for _, sample := range samples {
		_ = binary.Write(&buffer, binary.LittleEndian, int16(math.Round(float64(sample)*math.MaxInt16)))
	}
	return buffer.Bytes()
}
