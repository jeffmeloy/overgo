//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image/png"

	"overgo/internal/media"
)

// startLatent is where a rotary request's sampling starts (moss-ultra-sections/img2img): the noise as drawn where the
// request samples its whole schedule from noise alone, as before; otherwise as ComfyUI's flow models start a sampling
// (model_sampling CONST.noise_scaling), σ₀ times the noise plus 1 − σ₀ times the latent of the request's picture, or
// none, each product rounded to float32 before the sum.
func (g *Generator) startLatent(ctx context.Context, noise []float32) ([]float32, error) {
	if g.request.Image == "" && !g.request.cutsSchedule() {
		return noise, nil
	}
	var latent []float32
	if g.request.Image != "" {
		var err error
		if latent, err = sourceLatent(ctx, g.modelDir, g.request.Image, g.request.Width, g.request.Height); err != nil {
			return nil, err
		}
		if len(latent) != len(noise) {
			return nil, fmt.Errorf("rotary image: the picture's latent holds %d values; the noise %d", len(latent), len(noise))
		}
	}
	sigma := float32(g.schedule.Sigmas[0])
	rest := 1 - sigma
	start := make([]float32, len(noise))
	for index, value := range noise {
		start[index] = float32(sigma * value)
		if latent != nil {
			start[index] += float32(rest * latent[index])
		}
	}
	return start, nil
}

// sourceLatent is a request's picture as its sampling starts from it: a PNG, decoded, scaled to the canvas as ComfyUI's
// ImageScale scales it at lanczos with a centre crop, read in the signed unit range as ComfyUI's VAE reads it
// (2x - 1), encoded by the model's VAE to its patched, normalized latent, a token's channels together, and laid out
// as the denoiser reads a latent, each channel's tokens after another's. A picture other than a PNG is refused: the scaling is ComfyUI's to the byte
// for a PNG.
func sourceLatent(ctx context.Context, root, encoded string, width, height int) (_ []float32, err error) {
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("rotary image: the request's picture is not base64: %w", err)
	}
	picture, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("rotary image: the request's picture is not a PNG: %w", err)
	}
	levels, err := media.ScaleLikeComfyUI(picture, width, height)
	if err != nil {
		return nil, err
	}
	// The encoder reads a pixel's channels together, as the levels lie: each level over 255 in float32, as ComfyUI
	// holds the scaled picture, then 2x - 1.
	pixels := make([]float32, len(levels))
	for index, level := range levels {
		unit := float32(level) / media.UnitLevels
		pixels[index] = unit*2 - 1
	}
	encoder, err := NewPictureEncoder(ctx, root, height, width)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, encoder.Close(context.WithoutCancel(ctx))) }()
	rows, gridHeight, gridWidth, err := encoder.Encode(ctx, pixels)
	if err != nil {
		return nil, err
	}
	tokens := gridHeight * gridWidth
	if tokens == 0 || len(rows)%tokens != 0 {
		return nil, errors.New("rotary image: the picture's latent does not fill its grid")
	}
	channels := len(rows) / tokens
	planar := make([]float32, len(rows))
	for token := range tokens {
		for channel := range channels {
			planar[channel*tokens+token] = rows[token*channels+channel]
		}
	}
	return planar, nil
}
