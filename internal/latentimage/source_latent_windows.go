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
// none, each product rounded to float32 before the sum. A picture a schedule starts at σ₀ of 1 weighs nothing, so it is
// read, and refused as a picture or its crop would be, but not scaled or encoded (SQA-429, SQA-432): the image to
// image section's recording starts at 1, and its source encode built the VAE's encoder for a latent multiplied by
// zero, the start the noise either way.
func (g *Generator) startLatent(ctx context.Context, noise []float32) ([]float32, error) {
	if g.request.Image == "" && !g.request.cutsSchedule() {
		return noise, nil
	}
	sigma := float32(g.schedule.Sigmas[0])
	rest := 1 - sigma
	var latent []float32
	if g.request.Image != "" && rest == 0 {
		picture, err := requestPNG(g.request.Image, "picture")
		if err != nil {
			return nil, err
		}
		// Its crop is checked as its scaling would check it, without the scaling (SQA-432).
		bounds := picture.Bounds()
		if _, _, _, _, err := media.CropLikeComfyUI(bounds.Dx(), bounds.Dy(), g.request.Width, g.request.Height); err != nil {
			return nil, err
		}
	} else if g.request.Image != "" {
		var err error
		if latent, err = g.sections().sourceLatent(ctx, g.request.Image); err != nil {
			return nil, err
		}
		if len(latent) != len(noise) {
			return nil, fmt.Errorf("rotary image: the picture's latent holds %d values; the noise %d", len(latent), len(noise))
		}
	}
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
func (s *pictureSections) sourceLatent(ctx context.Context, encoded string) ([]float32, error) {
	picture, err := requestPNG(encoded, "picture")
	if err != nil {
		return nil, err
	}
	levels, err := media.ScaleLikeComfyUI(picture, s.width, s.height)
	if err != nil {
		return nil, err
	}
	unit := make([]float32, len(levels))
	for index, level := range levels {
		unit[index] = float32(level) / media.UnitLevels
	}
	return s.encodeUnit(ctx, unit)
}

// depthDirectory and depthCheckpoint are the folder of a MOSS Ultra root that holds the Depth Anything V2 Small
// checkpoint its depth sections estimate a picture's depth with, beside its enhancer, and the checkpoint's name there,
// as the workflow's DepthAnythingV2Preprocessor names it (612:438 ckpt_name).
const depthDirectory, depthCheckpoint = "depth", "depth_anything_v2_vits.pth"

// depthResolution is the detect resolution the workflow's preprocessor sizes its estimate to (612:438 resolution).
const depthResolution = 512

// sectionDepth is the depth map of a picture as the MOSS Ultra workflow's depth sections make it: the picture scaled
// to the canvas as ComfyUI's ImageScale scales it at lanczos with a centre crop (612:514), its depth estimated by the
// root's Depth Anything V2 Small at the detect resolution (612:438) on the card, a grey map at the estimate's size,
// white for near.
func (s *pictureSections) sectionDepth(ctx context.Context, picture image.Image) (*image.Gray, error) {
	levels, err := media.ScaleLikeComfyUI(picture, s.width, s.height)
	if err != nil {
		return nil, err
	}
	unit := make([]float32, len(levels))
	for index, level := range levels {
		unit[index] = float32(level) / media.UnitLevels
	}
	estimator, err := s.depthCard(ctx)
	if err != nil {
		return nil, err
	}
	depth, mapHeight, mapWidth, err := estimator.DepthMap(ctx, unit, s.height, s.width, depthResolution)
	if err != nil {
		return nil, err
	}
	gray := image.NewGray(image.Rect(0, 0, mapWidth, mapHeight))
	copy(gray.Pix, depth)
	return gray, nil
}

// controlDepth is the depth map a request's control samples by: its map, or the section's depth of its picture.
func (s *pictureSections) controlDepth(ctx context.Context, control DepthControl) (image.Image, error) {
	if control.Picture == "" {
		return requestPNG(control.Map, "depth map")
	}
	picture, err := requestPNG(control.Picture, "depth control's picture")
	if err != nil {
		return nil, err
	}
	return s.sectionDepth(ctx, picture)
}

// DepthPreview is the depth map a request's depth control samples by as the MOSS Ultra workflow previews it
// (PreviewImage 667): the map (controlDepth) scaled to the request's canvas as ComfyUI's ImageScale scales it at
// bilinear without a crop (612:439), written as ComfyUI writes a picture, 255 times each level truncated to 8 bits. The
// request is taken as its run takes it, its preset and canvas filling its size.
func DepthPreview(root string, request Request) (_ EncodedImage, err error) {
	filled, err := request.withPreset(CatalogIn(root))
	if err != nil {
		return EncodedImage{}, err
	}
	if filled.Control == nil {
		return EncodedImage{}, errors.New("latent image: the request holds no depth control to preview")
	}
	if err := filled.Control.Validate(); err != nil {
		return EncodedImage{}, err
	}
	if filled.Width <= 0 || filled.Height <= 0 {
		return EncodedImage{}, fmt.Errorf("latent image: a depth preview needs the request's size, not %d x %d", filled.Width, filled.Height)
	}
	sections := newPictureSections(root, filled.Width, filled.Height, nil)
	ctx := context.Background()
	defer func() { err = errors.Join(err, sections.close(ctx)) }()
	depth, err := sections.controlDepth(ctx, *filled.Control)
	if err != nil {
		return EncodedImage{}, err
	}
	bounds := depth.Bounds()
	shown := make([]float32, 0, bounds.Dx()*bounds.Dy())
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			shown = append(shown, float32(color.GrayModel.Convert(depth.At(x, y)).(color.Gray).Y)/media.UnitLevels)
		}
	}
	scaled, err := media.BilinearLikeComfyUI(shown, bounds.Dy(), bounds.Dx(), 1, filled.Height, filled.Width)
	if err != nil {
		return EncodedImage{}, err
	}
	preview := image.NewGray(image.Rect(0, 0, filled.Width, filled.Height))
	for index, value := range scaled {
		preview.Pix[index] = uint8(min(max(value*media.UnitLevels, 0), media.UnitLevels))
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, preview); err != nil {
		return EncodedImage{}, err
	}
	return EncodedImage{Data: encoded.Bytes(), MediaType: media.PNGMediaType, Channels: 1, Height: filled.Height, Width: filled.Width, Maximum: 1}, nil
}

// controlLatent is a request's depth control as its sampling reads it (moss-ultra-sections/depth-control), as ComfyUI's
// MOSSImageControl makes it: the map (controlDepth), its levels over 255 scaled to the canvas as ComfyUI's ImageScale
// scales it at bilinear without a crop, encoded (encodeUnit), after a colour map's latent of zeros, and the whole times
// the control's strength in float32.
func (s *pictureSections) controlLatent(ctx context.Context, control DepthControl) ([]float32, error) {
	depth, err := s.controlDepth(ctx, control)
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
	scaled, err := media.BilinearLikeComfyUI(unit, bounds.Dy(), bounds.Dx(), media.RGBChannels, s.height, s.width)
	if err != nil {
		return nil, err
	}
	latent, err := s.encodeUnit(ctx, scaled)
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
// the bytes its session takes: its picture to sample from and its depth control's map or picture, each a stored file,
// in base64.
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
		depth, err := read(r.Control.Map, "depth map")
		if err != nil {
			return Request{}, err
		}
		picture, err := read(r.Control.Picture, "depth control's picture")
		if err != nil {
			return Request{}, err
		}
		if depth != r.Control.Map || picture != r.Control.Picture {
			control := *r.Control
			control.Map, control.Picture = depth, picture
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
func (s *pictureSections) encodeUnit(ctx context.Context, unit []float32) ([]float32, error) {
	pixels := make([]float32, len(unit))
	for index, value := range unit {
		pixels[index] = value*2 - 1
	}
	encoder, err := s.pictureEncoder(ctx)
	if err != nil {
		return nil, err
	}
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
