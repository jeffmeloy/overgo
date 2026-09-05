package audioparity

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

func TestAudioArtifactQualificationMakesExactGraniteLoadable(t *testing.T) {
	election := qualificationElection(t)
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	inventory := qualificationInventory(t, election, recipe, modelartifact.TensorFormatSafetensors)
	root := qualificationAssets(t)
	qualification, err := QualifyAudioArtifact(t.Context(), election, inventory, root)
	if err != nil {
		t.Fatal(err)
	}
	if !qualification.Loadable || len(qualification.Passed) != len(qualificationChecks) || len(qualification.Refusals) != 0 {
		t.Fatalf("qualification = %+v", qualification)
	}
	if got := len(inventory.TensorInventory.Tensors); got != 550 {
		t.Fatalf("tensor requirements = %d, want 550", got)
	}
	input, found := inventory.TensorInventory.Tensor("encoder.input_linear.weight")
	if !found || len(input.Shape) != 2 || input.Shape[0] != 1024 || input.Shape[1] != 320 {
		t.Fatalf("input tensor = %+v", input)
	}

	batch, err := election.Batch("audio/qualification/exact-granite")
	if err != nil {
		t.Fatal(err)
	}
	if err := qualification.AugmentBatch(&batch, inventory, nil); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	loadable, found, err := store.ResolveAlias(t.Context(), qualification.Alias())
	if err != nil || !found || loadable != qualification.ID {
		t.Fatalf("loadable alias = (%s, %t, %v)", loadable, found, err)
	}
}

func TestAudioArtifactQualificationRejectsTensorShape(t *testing.T) {
	election := qualificationElection(t)
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	inventory := qualificationInventory(t, election, recipe, modelartifact.TensorFormatSafetensors)
	facts := inventory.TensorInventory.Tensors
	facts[0].Shape[0]++
	inventory.TensorInventory, err = modelartifact.NewTensorInventoryDocument(election.Model.ID, modelartifact.TensorFormatSafetensors, facts)
	if err != nil {
		t.Fatal(err)
	}
	qualification, err := QualifyAudioArtifact(t.Context(), election, inventory, qualificationAssets(t))
	if err != nil {
		t.Fatal(err)
	}
	if qualification.Loadable || !hasQualificationRefusal(qualification, "tensor-schema:") {
		t.Fatalf("qualification = %+v", qualification)
	}
}

func TestIncompatibleAudioGGUFQualification(t *testing.T) {
	election := qualificationElection(t)
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	inventory := qualificationInventory(t, election, recipe, modelartifact.TensorFormatGGUF)
	qualification, err := QualifyAudioArtifact(t.Context(), election, inventory, qualificationAssets(t))
	if err != nil {
		t.Fatal(err)
	}
	if qualification.Loadable || !hasQualificationRefusal(qualification, "container: parsed format \"gguf\"") {
		t.Fatalf("qualification = %+v", qualification)
	}
}

func qualificationElection(t *testing.T) Election {
	t.Helper()
	encoded, err := os.ReadFile(filepath.Join("testdata", "granite_speech_5_election.json"))
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	return election
}

func qualificationInventory(
	t *testing.T,
	election Election,
	recipe audioArchitectureRecipe,
	format modelartifact.TensorFormat,
) modelartifact.Inventory {
	t.Helper()
	facts := graniteTensorRequirements(recipe)
	for index := range facts {
		if facts[index].Shape == nil {
			facts[index].Shape = make([]uint64, 0)
		} else {
			facts[index].Shape = slices.Clone(facts[index].Shape)
		}
		facts[index].Storage = "bf16"
		facts[index].Bytes = 1
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(election.Model.ID, format, facts)
	if err != nil {
		t.Fatal(err)
	}
	return modelartifact.Inventory{Manifest: election.Model, TensorInventory: tensors}
}

func qualificationAssets(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	assets := map[string]string{
		"config.json": `{
  "architectures":["GraniteSpeech5ForCTC"], "model_type":"granite_speech5_ctc", "vocab_size":16384, "pad_token_id":0,
  "encoder_config":{"hidden_size":1024,"intermediate_size":4096,"num_hidden_layers":16,"num_attention_heads":8,"num_key_value_heads":8,"head_dim":128,"context_size":128,"conv_kernel_size":7,"conv_expansion_factor":2,"max_position_embeddings":512,"num_mel_bins":80,"subsample_layers":[0,1]}
}`,
		"preprocessor_config.json": `{"sample_rate":16000,"n_fft":512,"win_length":400,"hop_length":160,"n_mels":80,"stack_factor":2,"deltas":true,"delta_win_length":3,"logmel_floor_db":8}`,
		"processor_config.json":    `{"processor_class":"GraniteSpeech5Processor","feature_extractor":{"feature_extractor_type":"GraniteSpeech5FeatureExtractor","sampling_rate":16000,"num_mel_bins":80,"n_fft":512,"win_length":400,"hop_length":160}}`,
		"tokenizer_config.json":    `{"tokenizer_class":"ParakeetTokenizer","processor_class":"GraniteSpeech5Processor"}`,
		"tokenizer.json":           `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`,
	}
	for name, value := range assets {
		if err := os.WriteFile(filepath.Join(root, name), []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func hasQualificationRefusal(qualification AudioArtifactQualification, prefix string) bool {
	for _, refusal := range qualification.Refusals {
		if strings.HasPrefix(refusal, prefix) {
			return true
		}
	}
	return false
}
