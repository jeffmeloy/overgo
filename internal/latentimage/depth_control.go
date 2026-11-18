package latentimage

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
)

// DepthControl is a depth map a rotary request samples by (moss-ultra-sections/depth-control), as the MOSS Ultra
// workflow's MOSSImageControl applies one through the control adapter the request names (moss-control), whose embedder
// reads it beside the latent: the map scaled to the canvas as ComfyUI's ImageScale scales it at bilinear, encoded by the
// model's VAE, after a colour map's latent of zeros, times its strength, while the sampling's sigma lies within the
// share of the sampling from Start to End.
type DepthControl struct {
	// Picture is the picture whose depth map the control is, made as the MOSS Ultra workflow's depth sections make it
	// (sectionDepth): a PNG in base64 as JSON gives bytes, or, as a page names it, its stored identity, which the image
	// capability reads into its bytes (Request.ReadPictures).
	Picture string `json:"picture,omitzero" label:"picture" media:"image"`
	// Map is, in place of a picture, the depth map itself, white for near (cmd/depth makes one of a picture), given as
	// a picture is.
	Map string `json:"map,omitzero" label:"depth map" media:"image"`
	// Strength scales the map's latent: 0 is no control, 1 the adapter's trained strength (where unstated), at most 10.
	Strength *float32 `json:"strength,omitempty"`
	// Start and End are the shares of the sampling the control runs over, from its start: 0 and 1 where unstated.
	Start float64  `json:"start,omitzero"`
	End   *float64 `json:"end,omitempty"`
}

// largestControlStrength is the most a control's strength may be, as ComfyUI's node admits.
const largestControlStrength = 10

// Validate refuses a control without a map or a picture, or with both, of a strength outside 0..10, or whose share does
// not run from Start to End within the sampling.
func (c DepthControl) Validate() error {
	if (c.Map == "") == (c.Picture == "") {
		return errors.New("latent image: a depth control takes its map or its picture, one of them")
	}
	if strength := c.strength(); !checked.Finite32(strength) || strength < 0 || strength > largestControlStrength {
		return fmt.Errorf("latent image: a depth control's strength %g is outside 0..%d", strength, largestControlStrength)
	}
	if end := c.end(); !(c.Start >= 0 && c.Start <= end && end <= 1) {
		return fmt.Errorf("latent image: a depth control runs from %g to %g of the sampling; want 0 <= start <= end <= 1", c.Start, end)
	}
	return nil
}

func (c DepthControl) strength() float32 {
	if c.Strength == nil {
		return 1
	}
	return *c.Strength
}

func (c DepthControl) end() float64 {
	if c.End == nil {
		return 1
	}
	return *c.End
}

// window is the sigmas the control runs between, as ComfyUI's flow sampling turns its shares into sigmas
// (percent_to_sigma at the profile's shift): the sigma at End, low, and at Start, high.
func (c DepthControl) window(shift float64) (low, high float64) {
	return percentToSigma(c.end(), shift), percentToSigma(c.Start, shift)
}

// percentToSigma is ComfyUI's ModelSamplingDiscreteFlow.percent_to_sigma: 1 at the start, 0 at the end, and between
// them time_snr_shift(shift, 1 - percent).
func percentToSigma(percent, shift float64) float64 {
	switch {
	case percent <= 0:
		return 1
	case percent >= 1:
		return 0
	}
	t := 1 - percent
	if shift == 1 {
		return t
	}
	return shift * t / (1 + (shift-1)*t)
}
