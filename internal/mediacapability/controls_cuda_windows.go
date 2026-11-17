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
	controlProviders[modelrecipe.ModuleRoutedImagePrepare] = routedImageControls
}

// SenseNova-U1 draws, asked for nothing else, its app's first text-to-image resolution, 2048 x 2048, in 50 steps
// (SenseNova-U1/apps/comfyui: T2I_RESOLUTIONS "1:1", num_steps).
const routedImageSide, routedImageSteps = 2048, 50

// routedImageControls offers SenseNova's own defaults as the page's, and its presets as the preset control's choices.
func routedImageControls(_ string, controls []Control) ([]Control, error) {
	applyBounds(controls, map[string]Bounds{
		"width": {Default: routedImageSide}, "height": {Default: routedImageSide}, "steps": {Default: routedImageSteps, Step: 1},
	})
	presets, err := sensenovarecipe.PresetNames()
	if err != nil {
		return nil, err
	}
	return offerPresets(controls, senseNovaModel, presets)
}

// The request fields that name an image model's adapter and its strength (latentimage.Request.Adapter).
const (
	adapterNameControl     = "adapter"
	adapterStrengthControl = "adapter_strength"
)

// adapterControl says the control names an adapter or its strength.
func adapterControl(c Control) bool {
	return c.Name == adapterNameControl || c.Name == adapterStrengthControl
}

// rotaryControl says the control is one only a rotary profile takes, of the MOSS Ultra workflow's sections: a request's
// picture to sample from, which a page attaches, and its denoise (moss-ultra-sections/img2img), and prompt mode 2's
// enhancing by the enhancer a MOSS Ultra root holds (moss-ultra-sections/prompt-modes).
func rotaryControl(c Control) bool {
	return c.Name == "image" || c.Name == "denoise" || c.Name == "enhance"
}

func imageControls(directory string, controls []Control) ([]Control, error) {
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		return nil, err
	}
	presets, err := latentimage.PresetNames(directory)
	if err != nil {
		return nil, err
	}
	model, err := latentimage.ModelName(directory)
	if err != nil {
		return nil, err
	}
	if controls, err = offerPresets(controls, model, presets); err != nil {
		return nil, err
	}
	if profile.Rotary == nil {
		// A flow model offers the side its publisher draws (latentimage.PersonSize).
		side, err := latentimage.PersonSize(directory)
		if err != nil {
			return nil, err
		}
		applyBounds(controls, map[string]Bounds{
			"width": {Default: side}, "height": {Default: side}, "steps": {Default: profile.Sampling.DefaultSteps, Step: 1},
		})
		// It runs unguided, so it takes a negative prompt only where it honours one by normalized attention guidance.
		guided, err := latentimage.NegativeGuided(directory)
		if err != nil {
			return nil, err
		}
		return slices.DeleteFunc(controls, func(c Control) bool {
			return c.Name == "negative_prompt" && !guided || c.Name == "guidance_scale" || c.Name == "prompt_mode" || adapterControl(c) || rotaryControl(c)
		}), nil
	}
	// A rotary model offers the adapters it holds, applied unmerged at a strength of 1 unless the person states one,
	// and no adapter control where it holds none.
	adapters, err := latentimage.AdapterNames(directory)
	if err != nil {
		return nil, err
	}
	if len(adapters) == 0 {
		controls = slices.DeleteFunc(controls, adapterControl)
	}
	for index := range controls {
		if controls[index].Name == adapterNameControl {
			controls[index].Choices = adapters
		}
	}
	applyBounds(controls, map[string]Bounds{adapterStrengthControl: {Default: 1}})
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
	return controls, nil
}
