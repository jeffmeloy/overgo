//go:build windows

package mediacapability

import (
	"fmt"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/sensenovarecipe"
	"slices"
)

// The device-bound image requests exist only where their runtimes build.
func init() {
	requestTypes[modelrecipe.ModuleLatentImagePrepare] = latentimage.Request{}
	requestTypes[modelrecipe.ModuleRoutedImagePrepare] = sensenovarecipe.GenerationRequest{}
	controlProviders[modelrecipe.ModuleLatentImagePrepare] = imageControls
}

func imageControls(directory string, controls []Control) ([]Control, error) {
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		return nil, err
	}
	if profile.Rotary == nil {
		return slices.DeleteFunc(controls, func(c Control) bool {
			return c.Name == "negative_prompt" || c.Name == "guidance_scale" || c.Name == "prompt_mode"
		}), nil
	}
	policy := profile.Rotary
	if policy.DefaultSize <= 0 || float32(int(policy.Guidance)) != policy.Guidance {
		return nil, fmt.Errorf("image controls: source default extent or guidance cannot be represented")
	}
	controls = slices.DeleteFunc(controls, func(c Control) bool { return c.Name == "mu" || c.Name == "num_train_timesteps" })
	applyBounds(controls, map[string]Bounds{
		"width":          {Default: policy.DefaultSize, Step: policy.SpatialScale},
		"height":         {Default: policy.DefaultSize, Step: policy.SpatialScale},
		"steps":          {Default: profile.Sampling.DefaultSteps, Step: 1},
		"guidance_scale": {Default: int(policy.Guidance)},
	})
	for i := range controls {
		switch controls[i].Name {
		case "negative_prompt", "steps", "seed", "guidance_scale":
			controls[i].Advanced = true
		}
	}
	return controls, nil
}
