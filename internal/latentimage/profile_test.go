package latentimage

import (
	"math"
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
		profile.Sampling.dynamicShiftMu(profile.Sampling.MaxImageSequence, false) != 1.15 ||
		profile.Prompt.MaxTokens != 512 || profile.Prompt.PadToken != "<|endoftext|>" {
		t.Fatalf("resolved image policy = %+v", profile)
	}
	// The shift a request that names none is given. A checkpoint that is not
	// distilled takes one that grows with the picture: the base shift up to the
	// base sequence, the largest from the largest sequence on, and between them
	// in proportion. A distilled one takes the largest at every size, as the
	// model's own pipeline fixes it at 1.15 for its distilled checkpoint: a
	// picture of 512 by 512, 1024 patches, drawn at the shift its size alone
	// would give is not the picture the model's own code draws.
	sampling := profile.Sampling
	half := (sampling.BaseImageSequence + sampling.MaxImageSequence) / 2
	for _, shift := range []struct {
		sequence  int
		distilled bool
		want      float64
	}{
		{sampling.BaseImageSequence, false, sampling.BaseShift},
		{half, false, (sampling.BaseShift + sampling.MaxShift) / 2},
		{sampling.MaxImageSequence + 1, false, sampling.MaxShift},
		{sampling.BaseImageSequence, true, 1.15},
		{1024, true, 1.15},
		{sampling.MaxImageSequence, true, 1.15},
	} {
		if got := sampling.dynamicShiftMu(shift.sequence, shift.distilled); math.Abs(got-shift.want) > 1e-12 {
			t.Errorf("a picture of %d patches, distilled %t, is given a shift of %g, want %g", shift.sequence, shift.distilled, got, shift.want)
		}
	}
	content, err := profile.Content()
	if err != nil || content.Descriptor.ID != profile.ID {
		t.Fatalf("profile content = (%+v, %v)", content.Descriptor, err)
	}
}
