package media

import (
	"fmt"
	"strings"
)

// Preset is what a generated picture or clip ran, as its model resolved the request against its own defaults: the
// size, the frames of a clip, the denoising steps, the guidance scale, the sampler and the timestep shift. A time read
// from a generation says what it is a time of only with it: Wan takes 13 s at 32 x 16 in 2 steps and six minutes at
// 832 x 480 in 50.
type Preset struct {
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	Frames   int     `json:"frames,omitzero"`
	Steps    int     `json:"steps"`
	Guidance float64 `json:"guidance,omitzero"`
	Sampler  string  `json:"sampler"`
	Shift    float64 `json:"shift,omitzero"`
}

// Ran says the preset names a generation: one with no size, steps or sampler was resolved by none.
func (p Preset) Ran() bool { return p.Width > 0 && p.Height > 0 && p.Steps > 0 && p.Sampler != "" }

// String is the preset as a row prints it, "832 x 480, 81 frames, 50 steps, flow UniPC, guidance 5, shift 5", each part
// the generation has.
func (p Preset) String() string {
	parts := []string{fmt.Sprintf("%d x %d", p.Width, p.Height)}
	if p.Frames > 0 {
		parts = append(parts, fmt.Sprintf("%d frames", p.Frames))
	}
	parts = append(parts, fmt.Sprintf("%d steps", p.Steps), p.Sampler)
	for _, part := range []struct {
		name  string
		value float64
	}{{"guidance", p.Guidance}, {"shift", p.Shift}} {
		if part.value != 0 {
			parts = append(parts, fmt.Sprintf("%s %.4g", part.name, part.value))
		}
	}
	return strings.Join(parts, ", ")
}

// PresetOf is the preset a generation's output says it ran; an output that is no generation's says none.
func PresetOf(output any) Preset {
	if generated, ok := output.(interface{ RanPreset() Preset }); ok {
		return generated.RanPreset()
	}
	return Preset{}
}
