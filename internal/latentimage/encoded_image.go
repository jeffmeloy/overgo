package latentimage

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"

	"overgo/internal/artifact"
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

var encodedPNGContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: encodedImageMediaType, Schema: "overgo.encoded-image.png.v1",
}

// PNGContent validates and publishes the encoded PNG bytes without a JSON copy.
func PNGContent(image EncodedImage) (artifact.Content, error) {
	if image.MediaType != encodedImageMediaType || image.Channels != 3 || image.Height <= 0 || image.Width <= 0 {
		return artifact.Content{}, errors.New("latent image: invalid encoded PNG")
	}
	return encodedPNGContract.OwnedContentBytes(image.Data)
}

type rgbImage struct {
	pixels                     []float32
	width, height              int
	pixelStride, channelStride int
}

func (i rgbImage) ColorModel() color.Model { return color.RGBAModel }
func (i rgbImage) Bounds() image.Rectangle { return image.Rect(0, 0, i.width, i.height) }
func (i rgbImage) At(x, y int) color.Color {
	if x < 0 || x >= i.width || y < 0 || y >= i.height {
		return color.RGBA{}
	}
	offset := (y*i.width + x) * i.pixelStride
	return color.RGBA{
		R: pixelU8(i.pixels[offset]),
		G: pixelU8(i.pixels[offset+i.channelStride]),
		B: pixelU8(i.pixels[offset+2*i.channelStride]),
		A: 255,
	}
}

func pixelU8(value float32) uint8 {
	scaled := (float64(value) + 1) * 127.5
	return uint8(min(max(scaled, 0), 255))
}

func encodePNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, rgbImage{
		pixels: pixels, width: width, height: height, pixelStride: 3, channelStride: 1,
	})
}

// EncodePlanarPNG publishes normalized CHW RGB without an HWC copy.
func EncodePlanarPNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, rgbImage{
		pixels: pixels, width: width, height: height, pixelStride: 1, channelStride: width * height,
	})
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
	encoder := png.Encoder{CompressionLevel: png.DefaultCompression}
	if err := encoder.Encode(&encoded, source); err != nil {
		return EncodedImage{}, err
	}
	return EncodedImage{
		Data: encoded.Bytes(), MediaType: encodedImageMediaType,
		Channels: 3, Height: height, Width: width, Minimum: minimum, Maximum: maximum,
	}, nil
}
