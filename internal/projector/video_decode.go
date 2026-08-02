package projector

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"io"
)

func DecodeGIFVideo(reader io.Reader, maxFrames int) ([]image.Image, error) {
	if reader == nil {
		return nil, errors.New("projector: GIF reader is nil")
	}
	if maxFrames <= 0 {
		return nil, errors.New("projector: GIF frame limit must be positive")
	}
	decoded, err := gif.DecodeAll(reader)
	if err != nil {
		return nil, err
	}
	if len(decoded.Image) == 0 || decoded.Config.Width <= 0 || decoded.Config.Height <= 0 {
		return nil, errors.New("projector: GIF has no frames")
	}
	bounds := image.Rect(0, 0, decoded.Config.Width, decoded.Config.Height)
	background := color.Color(color.Transparent)
	if palette, ok := decoded.Config.ColorModel.(color.Palette); ok && int(decoded.BackgroundIndex) < len(palette) {
		background = palette[decoded.BackgroundIndex]
	}
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, image.NewUniform(background), image.Point{}, draw.Src)
	frames := make([]image.Image, 0, len(decoded.Image))
	for index, frame := range decoded.Image {
		if frame == nil || !frame.Bounds().In(bounds) {
			return nil, errors.New("projector: GIF frame geometry is invalid")
		}
		before := cloneRGBA(canvas)
		draw.Draw(canvas, frame.Bounds(), frame, frame.Bounds().Min, draw.Over)
		frames = append(frames, cloneRGBA(canvas))
		disposal := byte(gif.DisposalNone)
		if index < len(decoded.Disposal) {
			disposal = decoded.Disposal[index]
		}
		switch disposal {
		case gif.DisposalBackground:
			draw.Draw(canvas, frame.Bounds(), image.NewUniform(background), image.Point{}, draw.Src)
		case gif.DisposalPrevious:
			canvas = before
		}
	}
	return sampleVideoFrames(frames, maxFrames), nil
}

func sampleVideoFrames(frames []image.Image, maximum int) []image.Image {
	if len(frames) <= maximum {
		return frames
	}
	result := make([]image.Image, maximum)
	if maximum == 1 {
		result[0] = frames[0]
		return result
	}
	for index := range result {
		source := index * (len(frames) - 1) / (maximum - 1)
		result[index] = frames[source]
	}
	return result
}

func cloneRGBA(source *image.RGBA) *image.RGBA {
	result := image.NewRGBA(source.Bounds())
	copy(result.Pix, source.Pix)
	return result
}
