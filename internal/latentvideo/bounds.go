package latentvideo

import (
	"overgo/internal/tensor"
)

// ControlBound is what a page needs to offer presets for one numeric
// request field: the profile's default, the step a valid value moves by
// (the VAE stride times the patch size on that axis; zero when any
// value is valid) and, for a frame count, the frames one second holds.
type ControlBound struct {
	Default, Step, Rate int
}

// ProfileControlBounds derives the bounds of the Wan request's geometry
// from the profile's generation policy and the denoiser's patch size:
// width and height move by the spatial stride, frames by the temporal
// stride at the profile's sample rate, steps default without a stride.
func ProfileControlBounds(profile Profile, config DenoiserConfig) map[string]ControlBound {
	stride := func(axis int) int { return profile.Policy.VAEStride[axis] * config.PatchSize[axis] }
	return map[string]ControlBound{
		"width":  {Default: profile.Generation.Width, Step: stride(tensor.PairedExtent)},
		"height": {Default: profile.Generation.Height, Step: stride(tensor.SingletonExtent)},
		"frames": {Default: profile.Generation.Frames, Step: stride(tensor.FirstOffset), Rate: profile.SampleFPS},
		"steps":  {Default: profile.Generation.Steps},
	}
}

// ControlBounds resolves the profile and the denoiser config of the model
// at directory and derives the request's bounds from them.
func ControlBounds(directory string) (map[string]ControlBound, error) {
	profile, err := ResolveProfile(directory)
	if err != nil {
		return nil, err
	}
	config, err := LoadDenoiserConfig(directory, profile.Policy)
	if err != nil {
		return nil, err
	}
	return ProfileControlBounds(profile, config), nil
}
