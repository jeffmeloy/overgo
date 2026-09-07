package latentvideo

import (
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/tensor"
)

// TestProfileControlBoundsFollowTheStrides pins the presets' source: the
// shipped profile's generation policy gives the defaults, the VAE stride
// times the patch size gives the step per axis, and the sample rate rides
// the frame count.
func TestProfileControlBoundsFollowTheStrides(t *testing.T) {
	var catalog []Profile
	if err := strictjson.DecodeBytes(videoProfileCatalogJSON, &catalog); err != nil {
		t.Fatal(err)
	}
	profile := catalog[tensor.FirstOffset]
	var config DenoiserConfig
	config.Policy = profile.Policy
	config.PatchSize[tensor.FirstOffset], config.PatchSize[tensor.SingletonExtent], config.PatchSize[tensor.PairedExtent] = 1, 2, 2
	bounds := ProfileControlBounds(profile, config)
	want := map[string]ControlBound{
		"width":  {Default: profile.Generation.Width, Step: profile.Policy.VAEStride[tensor.PairedExtent] * 2},
		"height": {Default: profile.Generation.Height, Step: profile.Policy.VAEStride[tensor.SingletonExtent] * 2},
		"frames": {Default: profile.Generation.Frames, Step: profile.Policy.VAEStride[tensor.FirstOffset], Rate: profile.SampleFPS},
		"steps":  {Default: profile.Generation.Steps},
	}
	for name, expected := range want {
		if bounds[name] != expected {
			t.Fatalf("%s = %+v, want %+v", name, bounds[name], expected)
		}
	}
	if bounds["width"].Step == 0 || bounds["frames"].Rate == 0 || bounds["frames"].Default%bounds["frames"].Step != 1 {
		t.Fatalf("the shipped profile's geometry is not a valid preset source: %+v", bounds)
	}
}
