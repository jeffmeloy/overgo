package trainingworkflow

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelrecipe"
	"overgo/internal/peft"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// quantizedFixtureLayers is the depth of the Q8_0 base the quantized run trains over.
const quantizedFixtureLayers = 6

// TestQuantizedBaseAdapterRunMatchesTheDenseRun trains the same adapter over a Q8_0
// GGUF twice under one objective, once with every weight dequantized at load and
// once with the adapted projections held as stored: the two runs publish a verdict
// with the same held-out accuracies and leave the same adapter, to the last bit,
// and the quantized run's model is the base as stored with the adapter beside it
// in its checkpoint, which is what a quantized base can honestly publish.
func TestQuantizedBaseAdapterRunMatchesTheDenseRun(t *testing.T) {
	base := testutil.QuantizedDenseGGUF(t, quantizedFixtureLayers)
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	// A GGUF base registers against the store's architecture profiles.
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, text.store); err != nil {
		t.Fatal(err)
	}
	var adapters [2]peft.Adapter
	var accuracies [2][2]float64
	for index, quantized := range []bool{false, true} {
		output := filepath.Join(text.root, map[bool]string{false: "dense-gguf", true: "quantized-gguf"}[quantized])
		result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: base, OutputDirectory: output, Steps: 16, Host: true,
			LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1, Quantized: quantized},
		})
		if err != nil || !result.Verdict.Valid() {
			t.Fatalf("quantized=%t: run = (%+v, %v), want a verdict", quantized, result, err)
		}
		var verdict struct {
			BaselineValue  float64 `json:"baseline_value"`
			CandidateValue float64 `json:"candidate_value"`
		}
		content, found, err := artifact.ReadContent(ctx, text.store, result.Verdict)
		if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
			t.Fatalf("quantized=%t: verdict content = (found %t, %v)", quantized, found, err)
		}
		accuracies[index] = [2]float64{verdict.BaselineValue, verdict.CandidateValue}
		if adapters[index], err = peft.Load(filepath.Join(output, LoRAAdapterDirectory)); err != nil {
			t.Fatal(err)
		}
		t.Logf("quantized=%t: held-out token accuracy %.4f -> %.4f", quantized, verdict.BaselineValue, verdict.CandidateValue)
	}
	if accuracies[0] != accuracies[1] {
		t.Errorf("held-out accuracies differ: dense %v, quantized %v", accuracies[0], accuracies[1])
	}
	for name, module := range adapters[0].Modules {
		other := adapters[1].Modules[name]
		if !slices.Equal(module.Down, other.Down) || !slices.Equal(module.Up, other.Up) {
			t.Fatalf("%s: the quantized run left another adapter than the dense run", name)
		}
	}
	// The quantized run's model holds the projections as the base stores them: its trained shard
	// holds none of them, and its frozen shard is the base's own bytes.
	entries, err := os.ReadDir(filepath.Join(text.root, "quantized-gguf"))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	for _, want := range []string{trainingprogram.CheckpointModel, trainingprogram.CheckpointFrozenShard, LoRAAdapterDirectory} {
		if !slices.Contains(names, want) {
			t.Errorf("the quantized run's checkpoint holds %v, missing %s", names, want)
		}
	}
}

// TestQuantizedBaseOfALlamaPublishesTheBaseAsStored holds a quantized-base run over a
// llama GGUF to what it says it publishes: its query and key projections load
// reordered and in full precision, so the adapter merges into them in place, but the
// model the run publishes is the base as stored, every tensor with the base's type
// and bytes, the adapter beside it: a published model that held the update in
// some projections and not others would be served with the adapter applied twice
// to those.
func TestQuantizedBaseOfALlamaPublishesTheBaseAsStored(t *testing.T) {
	base := testutil.QuantizedLlamaGGUF(t, 2)
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, text.store); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(text.root, "llama-quantized")
	if _, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: base, OutputDirectory: output, Steps: 8, Host: true,
		LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1, Quantized: true},
	}); err != nil {
		t.Fatal(err)
	}
	original, err := gguf.Open(base)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	published, err := gguf.Open(filepath.Join(output, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	defer published.Close()
	for _, info := range original.Tensors {
		baseInfo, baseData := readStored(t, original, info.Name)
		publishedInfo, publishedData := readStored(t, published, info.Name)
		if publishedInfo.Type != baseInfo.Type || !bytes.Equal(publishedData, baseData) {
			t.Errorf("%s: published as %s, not the base's %s bytes", info.Name, publishedInfo.Type, baseInfo.Type)
		}
	}
}

// TestQuantizedBaseRefusesABaseThatIsNotQuantizedInAFile checks a safetensors base,
// which stores no quantization to keep, is refused rather than quietly trained dense.
func TestQuantizedBaseRefusesABaseThatIsNotQuantizedInAFile(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	_, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "refused"), Steps: 2, Host: true,
		LoRA: &LoRA{Rank: 2, Alpha: 4, Quantized: true},
	})
	if err == nil {
		t.Fatal("a quantized-base run over a safetensors directory was accepted")
	}
}
