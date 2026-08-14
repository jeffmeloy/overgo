package latentimage

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"
)

const encodedImageMediaType = "image/png"

// EncodedImage is a typed PNG output. Data owns the published artifact bytes.
type EncodedImage struct {
	Data      []byte  `json:"data"`
	MediaType string  `json:"media_type"`
	Channels  int     `json:"channels"`
	Height    int     `json:"height"`
	Width     int     `json:"width"`
	Minimum   float32 `json:"minimum"`
	Maximum   float32 `json:"maximum"`
}

type hwcRGB struct {
	pixels        []float32
	width, height int
}

type planarRGB struct {
	pixels        []float32
	width, height int
}

func (i hwcRGB) ColorModel() color.Model { return color.RGBAModel }
func (i hwcRGB) Bounds() image.Rectangle { return image.Rect(0, 0, i.width, i.height) }
func (i hwcRGB) At(x, y int) color.Color {
	if x < 0 || x >= i.width || y < 0 || y >= i.height {
		return color.RGBA{}
	}
	base := (y*i.width + x) * 3
	return color.RGBA{
		R: pixelU8(i.pixels[base]),
		G: pixelU8(i.pixels[base+1]),
		B: pixelU8(i.pixels[base+2]),
		A: 255,
	}
}

func (i planarRGB) ColorModel() color.Model { return color.RGBAModel }
func (i planarRGB) Bounds() image.Rectangle { return image.Rect(0, 0, i.width, i.height) }
func (i planarRGB) At(x, y int) color.Color {
	if x < 0 || x >= i.width || y < 0 || y >= i.height {
		return color.RGBA{}
	}
	plane := i.width * i.height
	offset := y*i.width + x
	return color.RGBA{
		R: pixelU8(i.pixels[offset]),
		G: pixelU8(i.pixels[plane+offset]),
		B: pixelU8(i.pixels[2*plane+offset]),
		A: 255,
	}
}

func pixelU8(value float32) uint8 {
	scaled := (float64(value) + 1) * 127.5
	return uint8(min(max(scaled, 0), 255))
}

func encodePNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, hwcRGB{pixels: pixels, width: width, height: height})
}

// EncodePlanarPNG publishes normalized CHW RGB without an HWC copy.
func EncodePlanarPNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, planarRGB{pixels: pixels, width: width, height: height})
}

func encodeRGB(pixels []float32, height, width int, source image.Image) (EncodedImage, error) {
	if height <= 0 || width <= 0 || len(pixels) != height*width*3 {
		return EncodedImage{}, errors.New("latent image: invalid RGB output")
	}
	minimum, maximum := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, value := range pixels {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return EncodedImage{}, errors.New("latent image: non-finite output")
		}
		minimum, maximum = min(minimum, value), max(maximum, value)
	}
	var encoded bytes.Buffer
	encoder := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := encoder.Encode(&encoded, source); err != nil {
		return EncodedImage{}, err
	}
	return EncodedImage{
		Data: encoded.Bytes(), MediaType: encodedImageMediaType,
		Channels: 3, Height: height, Width: width, Minimum: minimum, Maximum: maximum,
	}, nil
}
