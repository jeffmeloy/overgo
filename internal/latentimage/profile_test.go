package latentimage

import (
	"testing"

	"overgo/internal/testutil"
)

func TestImagePolicyComesFromRecipeProfile(t *testing.T) {
	dir := t.TempDir()
	testutil.WriteImageProfileFixture(t, dir)

	profile, err := ResolveProfile(dir)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Sampling.DefaultSteps != 8 || profile.Sampling.TrainTimesteps != 1000 ||
		profile.Sampling.dynamicShiftMu(profile.Sampling.MaxImageSequence) != 1.15 ||
		profile.Prompt.MaxTokens != 512 || profile.Prompt.PadToken != "<|endoftext|>" {
		t.Fatalf("resolved image policy = %+v", profile)
	}
	content, err := profile.Content()
	if err != nil || content.Descriptor.ID != profile.ID {
		t.Fatalf("profile content = (%+v, %v)", content.Descriptor, err)
	}
}
