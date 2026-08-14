package scratchmodel

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/adaptiveparity"
	"overgo/internal/artifact"
)

func loadOracle(t *testing.T) adaptiveparity.ScratchOracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	oracle, _, err := adaptiveparity.NormalizeScratchOracle(raw)
	if err != nil {
		t.Fatal(err)
	}
	return oracle
}

func TestScratchConstructionAuthority(t *testing.T) {
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps})
	if err != nil {
		t.Fatal(err)
	}
	wantDataset, err := oracle.CorpusID()
	if err != nil {
		t.Fatal(err)
	}
	wantSplit, err := oracle.SplitID()
	if err != nil {
		t.Fatal(err)
	}
	if construction.ID().Kind() != artifact.KindModel || construction.Dataset() != wantDataset || construction.SplitID() != wantSplit {
		t.Fatalf("construction identities = %s / %s / %s", construction.ID(), construction.Dataset(), construction.SplitID())
	}
	if construction.Authority().InitializedModel() != construction.ID() || construction.Authority().Dataset() != wantDataset {
		t.Fatal("training authority differs from construction")
	}
	split := construction.Split()
	if !reflect.DeepEqual(split.Train, oracle.Train) || !reflect.DeepEqual(split.Validation, oracle.Validation) || !reflect.DeepEqual(split.Test, oracle.Test) {
		t.Fatalf("split differs: %+v", split)
	}
	config := construction.Config()
	wantConfig := oracle.Config
	if config.VocabSize != wantConfig.VocabSize || config.BlockSize != wantConfig.BlockSize ||
		config.Embedding != wantConfig.Embedding || config.HeadDim != wantConfig.HeadDim ||
		config.HeadCount != wantConfig.Heads || config.LayerCount != wantConfig.Layers ||
		config.MLPWidth != wantConfig.MLPWidth || config.AttentionWindow != wantConfig.AttentionWindow ||
		config.BaseLR != wantConfig.BaseLearningRate || config.InitStd != wantConfig.Initialization ||
		config.Epsilon != wantConfig.Epsilon || config.BOS != wantConfig.BOS ||
		config.EstimatedParams != wantConfig.EstimatedParams ||
		!reflect.DeepEqual(config.Characters, wantConfig.Characters) ||
		!reflect.DeepEqual(config.CharacterIndex, wantConfig.CharacterToIndex) {
		t.Fatalf("derived config differs: %+v", config)
	}
	for index, document := range split.Train {
		tokens, err := construction.Tokens(document)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tokens, oracle.Tokens[index]) {
			t.Fatalf("tokens[%d] = %v, want %v", index, tokens, oracle.Tokens[index])
		}
	}
	parameters := construction.Parameters()
	if len(parameters) != len(oracle.Groups) {
		t.Fatalf("parameter groups = %d, want %d", len(parameters), len(oracle.Groups))
	}
	count := 0
	for index, parameter := range parameters {
		want := oracle.Groups[index]
		if parameter.Name != want.Name || parameter.Rows != want.Rows || parameter.Cols != want.Cols || parameter.Digest != want.WeightSHA256 {
			t.Fatalf("parameter %d differs: %+v", index, parameter)
		}
		wantInitializer := InitializerUniform
		if parameter.Name == "lt" || parameter.Name == "pos_bias" {
			wantInitializer = InitializerZero
		}
		if parameter.Role == "" || parameter.Initializer != wantInitializer || parameter.TiedTo != "" {
			t.Fatalf("parameter %q policy differs: %+v", parameter.Name, parameter)
		}
		count += parameter.Rows * parameter.Cols
	}
	if count != oracle.ParameterCount {
		t.Fatalf("parameter count = %d, want %d", count, oracle.ParameterCount)
	}
}

func TestScratchInitializedArtifactIdentity(t *testing.T) {
	oracle := loadOracle(t)
	facts := CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}
	first, err := Compile(facts)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(facts)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID() != second.ID() || first.Authority().ID() != second.Authority().ID() {
		t.Fatal("identical corpus facts changed construction identity")
	}
	config := first.Config()
	config.Characters[0] = "mutated"
	config.CharacterIndex["a"] = 99
	if first.Config().Characters[0] == "mutated" || first.Config().CharacterIndex["a"] == 99 {
		t.Fatal("construction exposed mutable config")
	}
	weights, ok := first.Weights("wte")
	if !ok {
		t.Fatal("wte absent")
	}
	weights[0]++
	stable, _ := first.Weights("wte")
	if weights[0] == stable[0] {
		t.Fatal("construction exposed mutable weights")
	}
	changed := facts
	changed.Documents = append([]string(nil), facts.Documents...)
	changed.Documents[0] += "a"
	third, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if third.ID() == first.ID() || third.Dataset() == first.Dataset() {
		t.Fatal("corpus mutation preserved initialized model identity")
	}
}
