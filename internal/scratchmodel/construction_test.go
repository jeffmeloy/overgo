package scratchmodel

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/adaptiveparity"
	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
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

func TestScratchProgramOwnsResidentExecution(t *testing.T) {
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	plan, program := construction.OptimizerPlan(), construction.Program()
	if plan.ParameterCount() != oracle.ParameterCount || plan.GroupCount() != len(oracle.Groups) {
		t.Fatalf("optimizer coverage=%d/%d want=%d/%d", plan.ParameterCount(), plan.GroupCount(), oracle.ParameterCount, len(oracle.Groups))
	}
	if program.OptimizerIdentity() != plan.Identity() || program.ID().Kind() != artifact.KindRecipe {
		t.Fatalf("program/optimizer authority=%s/%s", program.ID(), program.OptimizerIdentity())
	}
	operators := program.Operators()
	wantPhases := []trainingprogram.OperatorPhase{
		trainingprogram.PhaseBatch, trainingprogram.PhaseForward, trainingprogram.PhaseBackward,
		trainingprogram.PhaseOptimize, trainingprogram.PhaseEvaluate,
	}
	if len(operators) != len(wantPhases) {
		t.Fatalf("operator count=%d want=%d", len(operators), len(wantPhases))
	}
	for index, phase := range wantPhases {
		if operators[index].Phase != phase {
			t.Fatalf("operator %d phase=%s want=%s", index, operators[index].Phase, phase)
		}
	}
	for index := 0; index < plan.GroupCount(); index++ {
		group, _ := plan.Group(index)
		parameter := program.Parameters()[index]
		if group.Name != parameter.Name || group.Rows != parameter.Rows || group.Cols != parameter.Cols || !parameter.Trainable {
			t.Fatalf("shared group %d differs: %+v / %+v", index, group, parameter)
		}
	}

	pkg, err := build.Default.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range pkg.GoFiles {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.Contains(text, "func muonUpdate") || strings.Contains(text, "type value struct") || strings.Contains(text, "type hostState") {
			t.Fatalf("production scratch runtime owns forbidden primitive in %s", name)
		}
	}

	file, err := parser.ParseFile(token.NewFileSet(), "resident_cuda_windows.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundProgramRun, foundHardCoded := false, false
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != "Step" || function.Recv == nil {
			continue
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			foundProgramRun = foundProgramRun || selector.Sel.Name == "RunPhases"
			foundHardCoded = foundHardCoded || selector.Sel.Name == "forward" || selector.Sel.Name == "backward"
			return true
		})
	}
	if !foundProgramRun || foundHardCoded {
		t.Fatalf("resident Step program authority=%t hard-coded forward/backward=%t", foundProgramRun, foundHardCoded)
	}
}

func TestScratchConstructionAuthority(t *testing.T) {
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, AdaptiveDerivationProfile())
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
	config := construction.config
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

func TestAdaptiveDerivationProfileAuthority(t *testing.T) {
	oracle := loadOracle(t)
	facts := CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}
	profile := AdaptiveDerivationProfile()
	construction, err := Compile(facts, profile)
	if err != nil {
		t.Fatal(err)
	}
	wantProfile := profile.ID
	if construction.Authority().DerivationProfile() != wantProfile {
		t.Fatal("construction derivation profile differs")
	}
	if construction.config.Epsilon != profile.Epsilon || construction.config.MuonMomentum != profile.MuonMomentum {
		t.Fatal("construction ignored profile numerical policy")
	}
	changed := profile
	changed.Epsilon *= 2
	changed, err = NewDerivationProfile(changed)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(facts, changed)
	if err != nil {
		t.Fatal(err)
	}
	if second.config.Epsilon != changed.Epsilon || second.Authority().DerivationProfile() == wantProfile {
		t.Fatal("changed derivation profile did not change authority")
	}
	if _, err := Compile(facts, DerivationProfile{}); err == nil {
		t.Fatal("implicit derivation policy accepted")
	}
}

func TestScratchConstructionRejectsEmptyCorpus(t *testing.T) {
	for _, documents := range [][]string{{"", "", ""}, {"abc", "", "cab"}} {
		if _, err := Compile(CorpusFacts{Documents: documents, Seed: 7, Steps: 3}, AdaptiveDerivationProfile()); err == nil {
			t.Fatalf("empty corpus document accepted: %q", documents)
		}
	}
}

func TestScratchOracleRuntimeExcludedFromProduction(t *testing.T) {
	pkg, err := build.Default.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range pkg.GoFiles {
		if strings.Contains(name, "oracle") || strings.HasPrefix(name, "host_") {
			t.Fatalf("oracle runtime compiled into production: %s", name)
		}
	}
}

func TestScratchInitializedArtifactIdentity(t *testing.T) {
	oracle := loadOracle(t)
	facts := CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}
	first, err := Compile(facts, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(facts, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	if first.ID() != second.ID() || first.Authority().ID() != second.Authority().ID() {
		t.Fatal("identical corpus facts changed construction identity")
	}
	config := cloneConfig(first.config)
	config.Characters[0] = "mutated"
	config.CharacterIndex["a"] = 99
	if first.config.Characters[0] == "mutated" || first.config.CharacterIndex["a"] == 99 {
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
	third, err := Compile(changed, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	if third.ID() == first.ID() || third.Dataset() == first.Dataset() {
		t.Fatal("corpus mutation preserved initialized model identity")
	}
}
