package trainingworkflow

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

// averageDocuments is the text a tiny model trains on in the averaged runs: twelve
// prefixes of a repeating pattern, each its own length.
func averageDocuments() []string {
	var documents []string
	for index := range 12 {
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	return documents
}

// TestAveragedTrainingPublishesTheAverage runs the same training twice, once
// with a moving average of the weights and once without: the trained model the
// first publishes is not the one the second does, each judged on the held-out
// records, and the second is what the run published before the average existed.
func TestAveragedTrainingPublishesTheAverage(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	var weights [2][]byte
	for index, decay := range []float64{0, 0.9} {
		output := filepath.Join(text.root, map[float64]string{0: "raw", 0.9: "averaged"}[decay])
		result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: 12, Host: true, Average: decay,
		})
		if err != nil || !result.Verdict.Valid() {
			t.Fatalf("average %v: run = (%+v, %v), want a verdict", decay, result, err)
		}
		path, err := trainingprogram.CheckpointWeightsPath(output)
		if err != nil {
			t.Fatal(err)
		}
		if weights[index], err = os.ReadFile(path); err != nil {
			t.Fatal(err)
		}
	}
	if string(weights[0]) == string(weights[1]) {
		t.Fatal("the averaged run published the weights a run without an average does")
	}
}

// TestAveragedAdapterMergesToThePublishedModel trains an adapter under an average
// and checks the model the run published is the base with that adapter merged
// in, the adapter in its PEFT directory being the average too: the file means
// what was judged.
func TestAveragedAdapterMergesToThePublishedModel(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	output := filepath.Join(text.root, "averaged-adapter")
	if _, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: 24, Host: true, Average: 0.9,
		LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1},
	}); err != nil {
		t.Fatal(err)
	}
	adapter, err := peft.Load(filepath.Join(output, LoRAAdapterDirectory))
	if err != nil {
		t.Fatal(err)
	}
	merged := filepath.Join(t.TempDir(), "merged")
	if _, err := peft.Merge(adapter, text.model, merged); err != nil {
		t.Fatal(err)
	}
	fromAdapter, err := safetensors.OpenSource(merged)
	if err != nil {
		t.Fatal(err)
	}
	defer fromAdapter.Close()
	fromCheckpoint, err := safetensors.OpenSource(output)
	if err != nil {
		t.Fatal(err)
	}
	defer fromCheckpoint.Close()
	for name, tensor := range fromCheckpoint.Tensors {
		got, err := safetensors.ReadF32(fromAdapter.Tensors[name])
		if err != nil {
			t.Fatal(err)
		}
		want, err := safetensors.ReadF32(tensor)
		if err != nil {
			t.Fatal(err)
		}
		for index := range want {
			if math.Abs(float64(got[index]-want[index])) > 1e-5 {
				t.Fatalf("%s[%d]: merged adapter %v, published model %v", name, index, got[index], want[index])
			}
		}
	}
}

// TestAveragedTrainingRefusesWhatItCannotAverage checks a decay outside (0, 1)
// and an average on the device trainer, which has none, are refused before
// anything trains.
func TestAveragedTrainingRefusesWhatItCannotAverage(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	for name, request := range map[string]TextHeldoutRequest{
		"decay of one":  {Host: true, Average: 1},
		"negative":      {Host: true, Average: -0.5},
		"on the device": {Host: false, Average: 0.9},
	} {
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps = text.alias, text.model, filepath.Join(text.root, "refused-"+strings.ReplaceAll(name, " ", "-")), 2
		if _, err := RunTextHeldout(t.Context(), text.store, request); err == nil || !strings.Contains(err.Error(), "moving average") {
			t.Errorf("%s: err = %v, want the refusal naming the moving average", name, err)
		}
	}
}

// TestAveragedAdapterIsNotTheRawAdapter checks the adapter an averaged run leaves
// is the average of its factors, not the factors the last update left: a run
// under the same seed and steps without the average leaves different ones.
func TestAveragedAdapterIsNotTheRawAdapter(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	var adapters [2]peft.Adapter
	for index, decay := range []float64{0, 0.9} {
		output := filepath.Join(text.root, map[float64]string{0: "raw-adapter", 0.9: "averaged-adapter"}[decay])
		if _, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: 24, Host: true, Average: decay,
			LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1},
		}); err != nil {
			t.Fatal(err)
		}
		adapter, err := peft.Load(filepath.Join(output, LoRAAdapterDirectory))
		if err != nil {
			t.Fatal(err)
		}
		adapters[index] = adapter
	}
	for name, raw := range adapters[0].Modules {
		for position, value := range raw.Up {
			if value != adapters[1].Modules[name].Up[position] {
				return
			}
		}
	}
	t.Fatal("the averaged adapter's factors equal the raw adapter's")
}
