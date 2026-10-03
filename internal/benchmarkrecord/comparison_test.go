package benchmarkrecord

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

func TestBenchmarkComparisonRecord(t *testing.T) {
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	recipeID, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	baseline := BenchmarkConfiguration{Tokens: 4, Runs: 3, Warmup: 0}
	candidate := baseline
	candidate.CachePrompt = true
	baseDigest, err := baseline.Digest()
	if err != nil {
		t.Fatal(err)
	}
	candidateDigest, err := candidate.Digest()
	if err != nil || baseDigest == candidateDigest {
		t.Fatalf("options digest did not bind the changed factor: %v", err)
	}
	prompt := strings.Repeat("a", 64)
	tokens := strings.Repeat("b", 64)
	observation := func(total float64) BenchmarkObservation {
		return BenchmarkObservation{PromptDigest: prompt, TokenDigests: []string{tokens},
			PromptTokens: 2, HostLogitTokens: 4, OutputTokens: 4, TotalMilliseconds: total, TTFTMilliseconds: 5,
			DecodeMilliseconds: total - 5, KernelLaunches: 12, HostToDeviceBytes: 64}
	}
	input := BenchmarkComparisonRecord{
		Identity: BenchmarkIdentity{Model: model, Recipe: recipeID, TokenizerContainer: model,
			Environment: environment, CodeCommit: strings.Repeat("c", 40),
			ModuleDigest: strings.Repeat("d", 64), WorkloadDigest: strings.Repeat("e", 64),
			PromptDigests: []string{prompt}, RealizedResidency: recipe.RealizedDeviceNative},
		Factor: "cache_prompt", Baseline: baseline, Candidate: candidate,
		Exclusive: true, SharedLoadMilliseconds: 200, CacheUpperBound: true,
		Pairs: []BenchmarkPair{
			{Index: 0, First: "baseline", Baseline: observation(100), Candidate: observation(90)},
			{Index: 1, First: "candidate", Baseline: observation(110), Candidate: observation(100)},
			{Index: 2, First: "baseline", Baseline: observation(120), Candidate: observation(110)},
		},
	}
	protocol, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("greedy/continue-after-eog/device-native"))
	contract, err := input.Identity.Comparison(baseline, protocol)
	if err != nil || !contract.Valid() {
		t.Fatalf("measurement contract = %s, %v", contract, err)
	}
	for name, change := range map[string]func(*BenchmarkIdentity, *BenchmarkConfiguration, *artifact.ID){
		"model": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.Model, _ = artifact.IdentifyBytes(artifact.KindModel, []byte("other model"))
			identity.TokenizerContainer = identity.Model
		},
		"recipe": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.Recipe, _ = artifact.IdentifyBytes(artifact.KindRecipe, []byte("other recipe"))
		},
		"environment": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.Environment, _ = artifact.IdentifyBytes(artifact.KindEvidence, []byte("other environment"))
		},
		"producer": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.CodeCommit = strings.Repeat("f", 40)
		},
		"binary": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.ModuleDigest = strings.Repeat("f", 64)
		},
		"workload": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.WorkloadDigest = strings.Repeat("f", 64)
		},
		"residency": func(identity *BenchmarkIdentity, _ *BenchmarkConfiguration, _ *artifact.ID) {
			identity.RealizedResidency = recipe.RealizedHostCache
		},
		"options": func(_ *BenchmarkIdentity, configuration *BenchmarkConfiguration, _ *artifact.ID) {
			configuration.CachePrompt = true
		},
		"protocol": func(_ *BenchmarkIdentity, _ *BenchmarkConfiguration, protocol *artifact.ID) {
			*protocol, _ = artifact.IdentifyBytes(artifact.KindProfile, []byte("other protocol"))
		},
		"unknown": func(_ *BenchmarkIdentity, _ *BenchmarkConfiguration, protocol *artifact.ID) {
			*protocol = artifact.ID{}
		},
	} {
		t.Run("contract/"+name, func(t *testing.T) {
			identity, configuration, changedProtocol := input.Identity, baseline, protocol
			change(&identity, &configuration, &changedProtocol)
			changed, err := identity.Comparison(configuration, changedProtocol)
			if name == "unknown" {
				if err == nil {
					t.Fatal("unknown contract accepted")
				}
			} else if err != nil || changed == contract {
				t.Fatalf("changed %s compared on the same contract: %s, %v", name, changed, err)
			}
		})
	}
	record, err := NewBenchmarkComparisonRecord(input)
	if err != nil {
		t.Fatal(err)
	}
	if !record.ID.Valid() || record.Outcome != "all-pairs-faster" || record.MedianDeltaMilliseconds != -10 {
		t.Fatalf("comparison verdict = %+v", record)
	}
	content, err := record.Content()
	if err != nil {
		t.Fatal(err)
	}
	replay, err := benchmarkComparisonCodec.Parse(content.Data)
	if err != nil || replay.ID != record.ID || replay.Outcome != record.Outcome {
		t.Fatalf("comparison replay = %+v, %v", replay, err)
	}
	if _, err := record.Batch("benchmarks/fixture"); err != nil {
		t.Fatal(err)
	}
	negative := cloneBenchmarkComparison(input)
	for index := range negative.Pairs {
		negative.Pairs[index].Candidate.TotalMilliseconds = negative.Pairs[index].Baseline.TotalMilliseconds + 10
	}
	regression, err := NewBenchmarkComparisonRecord(negative)
	if err != nil || regression.Outcome != "all-pairs-slower" || len(regression.Pairs) != 3 {
		t.Fatalf("negative result was lost: %+v, %v", regression, err)
	}
	noGain := cloneBenchmarkComparison(input)
	for index := range noGain.Pairs {
		noGain.Pairs[index].Candidate.TotalMilliseconds = noGain.Pairs[index].Baseline.TotalMilliseconds
	}
	tied, err := NewBenchmarkComparisonRecord(noGain)
	if err != nil || tied.Outcome != "tie" || tied.MedianDeltaMilliseconds != 0 {
		t.Fatalf("no-gain result was lost: %+v, %v", tied, err)
	}
	one := cloneBenchmarkComparison(input)
	one.Pairs = one.Pairs[:1]
	one.Baseline.Runs, one.Candidate.Runs = 1, 1
	single, err := NewBenchmarkComparisonRecord(one)
	if err != nil || single.Outcome != "all-pairs-faster" || len(single.Pairs) != 1 {
		t.Fatalf("single descriptive pair was lost: %+v, %v", single, err)
	}
	for name, change := range map[string]func(*BenchmarkComparisonRecord){
		"wrong output":      func(v *BenchmarkComparisonRecord) { v.Pairs[1].Candidate.TokenDigests[0] = strings.Repeat("f", 64) },
		"wrong order":       func(v *BenchmarkComparisonRecord) { v.Pairs[1].First = "baseline" },
		"other factor":      func(v *BenchmarkComparisonRecord) { v.Candidate.Speculative = true },
		"other prompt":      func(v *BenchmarkComparisonRecord) { v.Pairs[2].Candidate.PromptDigest = strings.Repeat("f", 64) },
		"unexclusive":       func(v *BenchmarkComparisonRecord) { v.Exclusive = false },
		"legacy residency":  func(v *BenchmarkComparisonRecord) { v.Identity.RealizedResidency = "" },
		"false cache label": func(v *BenchmarkComparisonRecord) { v.CacheUpperBound = false },
	} {
		t.Run(name, func(t *testing.T) {
			changed := cloneBenchmarkComparison(input)
			changed.Pairs = slices.Clone(changed.Pairs)
			change(&changed)
			if _, err := NewBenchmarkComparisonRecord(changed); err == nil {
				t.Fatal("incomparable record accepted")
			}
		})
	}
}
