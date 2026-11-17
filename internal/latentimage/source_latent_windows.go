//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"

	"overgo/internal/artifact"
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
// ImageScale scales it at lanczos with a centre crop, its levels over 255 in float32 as ComfyUI holds the scaled
// picture, and encoded (encodeUnit). A picture other than a PNG is refused: the scaling is ComfyUI's to the byte for a
// PNG.
func sourceLatent(ctx context.Context, root, encoded string, width, height int) ([]float32, error) {
	picture, err := requestPNG(encoded, "picture")
	if err != nil {
		return nil, err
	}
	levels, err := media.ScaleLikeComfyUI(picture, width, height)
	if err != nil {
		return nil, err
	}
	unit := make([]float32, len(levels))
	for index, level := range levels {
		unit[index] = float32(level) / media.UnitLevels
	}
	return encodeUnit(ctx, root, unit, width, height)
}

// controlLatent is a request's depth control as its sampling reads it (moss-ultra-sections/depth-control), as ComfyUI's
// MOSSImageControl makes it: the map, a PNG, its levels over 255 scaled to the canvas as ComfyUI's ImageScale scales
// it at bilinear without a crop, encoded (encodeUnit), after a colour map's latent of zeros, and the whole times the
// control's strength in float32.
func controlLatent(ctx context.Context, root string, control DepthControl, width, height int) ([]float32, error) {
	depth, err := requestPNG(control.Map, "depth map")
	if err != nil {
		return nil, err
	}
	bounds := depth.Bounds()
	unit := make([]float32, 0, bounds.Dx()*bounds.Dy()*media.RGBChannels)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := color.NRGBAModel.Convert(depth.At(x, y)).(color.NRGBA)
			unit = append(unit, float32(pixel.R)/media.UnitLevels, float32(pixel.G)/media.UnitLevels, float32(pixel.B)/media.UnitLevels)
		}
	}
	scaled, err := media.BilinearLikeComfyUI(unit, bounds.Dy(), bounds.Dx(), media.RGBChannels, height, width)
	if err != nil {
		return nil, err
	}
	latent, err := encodeUnit(ctx, root, scaled, width, height)
	if err != nil {
		return nil, err
	}
	strength := control.strength()
	out := make([]float32, 2*len(latent))
	for index, value := range latent {
		out[len(latent)+index] = value * strength
	}
	return out, nil
}

// ReadPictures is the request with the pictures it names by their stored identity, as a page attaches them, read into
// the bytes its session takes: its picture to sample from and its depth control's map, each a stored file, in base64.
// A picture the request gives as its bytes stays as it came, and so does a request naming none.
func (r Request) ReadPictures(ctx context.Context, store artifact.Reader) (Request, error) {
	read := func(named, which string) (string, error) {
		id, err := artifact.ParseID(named)
		if err != nil {
			return named, nil
		}
		if id.Kind() != artifact.KindFile {
			return "", fmt.Errorf("latent image: the request's %s names %s, not a stored file", which, id)
		}
		content, present, err := artifact.ReadContent(ctx, store, id)
		if err != nil {
			return "", err
		}
		if !present {
			return "", fmt.Errorf("latent image: the request's %s, %s, is not stored", which, id)
		}
		return base64.StdEncoding.EncodeToString(content.Data), nil
	}
	var err error
	if r.Image, err = read(r.Image, "picture"); err != nil {
		return Request{}, err
	}
	if r.Control != nil {
		named, err := read(r.Control.Map, "depth map")
		if err != nil {
			return Request{}, err
		}
		if named != r.Control.Map {
			control := *r.Control
			control.Map = named
			r.Control = &control
		}
	}
	return r, nil
}

// requestPNG is a PNG a request carries in base64, which of the request's pictures it is named in a refusal.
func requestPNG(encoded, which string) (image.Image, error) {
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("rotary image: the request's %s is not base64: %w", which, err)
	}
	picture, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("rotary image: the request's %s is not a PNG: %w", which, err)
	}
	return picture, nil
}

// encodeUnit is a picture of height by width RGB values in the unit range, a pixel's channels together, encoded as
// ComfyUI's VAE encodes it: read in the signed unit range (2x - 1), encoded by the model's VAE to its patched,
// normalized latent, a token's channels together, and laid out as the denoiser reads a latent, each channel's tokens
// after another's.
func encodeUnit(ctx context.Context, root string, unit []float32, width, height int) (_ []float32, err error) {
	pixels := make([]float32, len(unit))
	for index, value := range unit {
		pixels[index] = value*2 - 1
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
