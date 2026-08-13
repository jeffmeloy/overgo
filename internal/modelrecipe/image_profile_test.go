package modelrecipe

import "testing"

func TestImagePolicyComesFromRecipeProfile(t *testing.T) {
	profile, ok, err := ImageProfileForPipeline("Krea2Pipeline")
	if err != nil {
		t.Fatal(err)
	}
	if !ok || profile.ID != "latent.krea2" || profile.Conditioning.MaxPromptTokens != 512 ||
		profile.Sampling.Steps != 8 || profile.Sampling.NumTrainTimesteps != 1000 || profile.Sampling.DynamicShiftMu != 1.15 {
		t.Fatalf("profile=%+v present=%v", profile, ok)
	}
	if _, ok, err := ImageProfileForPipeline("unknown"); err != nil || ok {
		t.Fatalf("unknown profile present=%v err=%v", ok, err)
	}
}
