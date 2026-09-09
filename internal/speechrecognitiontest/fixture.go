// Package speechrecognitiontest owns the captured small-encoder consumer fixture.
package speechrecognitiontest

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

// Fixture binds a real small encoder and its explicit test admission policy.
type Fixture struct {
	Definition recipe.Definition
	Policy     dataset.AudioInspectionPolicy
}

// Publish registers captured weights, tokenizer, frontend and an executable recipe.
// The fixture runs inference; it does not inject recorded predictions.
func Publish(t *testing.T, store artifact.Repository, oraclePath string) Fixture {
	t.Helper()
	data, err := os.ReadFile(oraclePath)
	if err != nil {
		t.Fatal(err)
	}
	var captured struct {
		Declaration speechrecognition.Declaration `json:"declaration"`
		Weights     map[string]struct {
			Shape  []int     `json:"shape"`
			Values []float32 `json:"values"`
		} `json:"weights"`
	}
	if err := json.Unmarshal(data, &captured); err != nil {
		t.Fatal(err)
	}
	values, shapes := make(map[string][]float32), make(map[string][]int)
	for name, weight := range captured.Weights {
		values[name], shapes[name] = weight.Values, weight.Shape
	}
	directory := t.TempDir()
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"config.json":    `{"model_type":"fixture"}`,
		"tokenizer.json": `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"x":3,"y":4},"merges":[],"byte_fallback":false}}`,
	} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	frontend := audiodsp.FrontendConfig{
		SampleRate: 16000, Geometry: recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 2, FeatureBins: 4},
		FFTLength: 16, FrameSpan: 4, Padding: "zero", Window: "rectangular",
		Mel: audiodsp.MelConfig{Scale: "htk", MinFrequency: 20, MaxFrequency: 7000},
		Log: audiodsp.LogConfig{Power: true, Base: "natural", GuardMode: "clamp", Guard: 1e-8, Scale: 1},
	}
	profile, err := speechrecognition.NewExecutionProfile(speechrecognition.ExecutionProfile{Frontend: frontend,
		Grouping: audiodsp.GroupedFeatureConfig{StackFrames: 1, FinalFrameSamples: 4}, Encoder: captured.Declaration, BlankToken: 0, Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	definition := PublishModel(t, store, directory, profile)
	return Fixture{Definition: definition, Policy: dataset.AudioInspectionPolicy{
		MaximumEncodedBytes: 1 << 20, MaximumSamples: 1 << 16, ClipThreshold: .999,
		Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1,
			SilenceRMSThreshold: .001, MaximumAbsoluteDCOffset: .1},
	}}
}

// PublishModel registers an existing model directory and exact execution profile.
// Source weights remain in place; the shared fixture owner publishes identities.
func PublishModel(t *testing.T, store artifact.Repository, directory string, profile speechrecognition.ExecutionProfile) recipe.Definition {
	t.Helper()
	hf, err := hfrepo.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromHFRepository(hf)
	if closeErr := hf.Close(); err != nil || closeErr != nil {
		t.Fatalf("inventory: %v, close: %v", err, closeErr)
	}
	commit := func(batch artifact.Batch, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit(inventory.Batch("transcription-fixture/model/" + inventory.Manifest.ID.String()))
	var tokenizer artifact.ID
	for _, component := range inventory.Manifest.Components {
		if component.Name == "tokenizer.json" {
			tokenizer = component.Artifact
		}
	}
	contract, err := modelrecipe.NewAudioContract(
		recipecontract.AudioFormat{SampleRate: uint64(profile.Frontend.SampleRate), Channels: 1, Encoding: "pcm-f32le"}, profile.Frontend.Geometry, artifact.ID{}, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	commit(profile.Batch("transcription-fixture/profile/" + profile.ID.String()))
	commit(contract.Batch("transcription-fixture/contract/" + contract.ID.String()))
	definition, err := modelrecipe.TranscriptionDefinition(inventory.Manifest.ID, contract.ID, profile.ID, tokenizer, inventory.TensorInventory.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "transcription-fixture/recipe/"+definition.ID.String(), definition); err != nil {
		t.Fatal(err)
	}
	return definition
}

// Wave returns deterministic PCM for the fixture's speech or silent control.
func Wave(t *testing.T, silent bool) ([]byte, uint64) {
	t.Helper()
	samples := make([]int16, 128)
	if !silent {
		for i := range samples {
			samples[i] = int16(8192 * math.Sin(2*math.Pi*float64(i)/16))
		}
	}
	return testutil.MonoPCM16WAV(16000, samples), uint64(len(samples))
}
