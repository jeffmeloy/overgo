package scratchmodel

import (
	"reflect"
	"testing"

	"overgo/internal/model"
)

// TestConstructionEmitsExecutableProfile: constructed profile remains executable authority.
func TestConstructionEmitsExecutableProfile(t *testing.T) {
	construction, err := Compile(
		CorpusFacts{Documents: bridgeCorpus(), Seed: 7, Steps: 2},
		testDerivationProfile(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	profile := construction.Architecture()
	if profile.Name != ScratchArchitecture {
		t.Fatalf("emitted architecture %q, want %q", profile.Name, ScratchArchitecture)
	}
	registered, ok := model.LookupArchitecture(ScratchArchitecture)
	if !ok {
		t.Fatalf("architecture %q is not registered in the executor vocabulary", ScratchArchitecture)
	}
	if !reflect.DeepEqual(registered, profile) {
		t.Fatal("emitted profile differs from the registered executor profile")
	}
	if err := model.ValidateArchitectureProfile(profile); err != nil {
		t.Fatalf("emitted profile fails the shared validation authority: %v", err)
	}
	if profile.Normalization != model.NormalizationMAD ||
		profile.FeedForward != model.FeedForwardReLU ||
		profile.Position != model.PositionLearnedAbsolute {
		t.Fatalf("profile policies %+v do not describe the constructed topology", profile)
	}
	tokens, err := construction.Tokens(construction.Split().Train[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := construction.CompileForwardGraph(tokens); err != nil {
		t.Fatalf("policy-driven forward refused the registered profile: %v", err)
	}
	misdescribed := construction
	misdescribed.architecture.Normalization = model.NormalizationRMS
	if _, err := misdescribed.CompileForwardGraph(tokens); err == nil {
		t.Fatal("forward compiled under a profile that misdescribes the topology")
	}
}

func bridgeCorpus() []string {
	return []string{
		"the quick brown fox jumps over the lazy dog",
		"pack my box with five dozen liquor jugs",
		"how vexingly quick daft zebras jump",
		"sphinx of black quartz judge my vow",
	}
}
