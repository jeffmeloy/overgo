package media

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"
)

const (
	// PNGMediaType identifies Portable Network Graphics image content.
	PNGMediaType       = "image/png"
	RGBChannels        = 3
	RasterSampleCenter = 0.5
)

// ValidRGBRaster reports whether an encoded raster carries valid RGB geometry.
func ValidRGBRaster(mediaType string, channels, height, width int) bool {
	return mediaType == PNGMediaType && channels == RGBChannels && height > 0 && width > 0
}

func AffineRGB(values []float32, scale, bias [RGBChannels]float32) []float32 {
	output := make([]float32, len(values))
	for index, value := range values {
		channel := index % RGBChannels
		output[index] = value*scale[channel] + bias[channel]
	}
	return output
}

type normalizedRGBImage struct {
	pixels                     []float32
	width, height              int
	pixelStride, channelStride int
}

// ColorModel returns the standard RGBA color model.
func (i normalizedRGBImage) ColorModel() color.Model { return color.RGBAModel }

// Bounds returns the image pixel rectangle.
func (i normalizedRGBImage) Bounds() image.Rectangle { return image.Rect(0, 0, i.width, i.height) }

// At returns the normalized pixel at the requested coordinate.
func (i normalizedRGBImage) At(x, y int) color.Color {
	if x < 0 || x >= i.width || y < 0 || y >= i.height {
		return color.RGBA{}
	}
	offset := (y*i.width + x) * i.pixelStride
	return color.RGBA{
		R: normalizedPixel8(i.pixels[offset]),
		G: normalizedPixel8(i.pixels[offset+i.channelStride]),
		B: normalizedPixel8(i.pixels[offset+(RGBChannels-1)*i.channelStride]),
		A: ^uint8(0),
	}
}

func normalizedPixel8(value float32) uint8 {
	magnitude := float64(^uint8(0))
	scaled := (float64(value) + 1) * magnitude / 2
	return uint8(min(max(scaled, 0), magnitude))
}

// EncodeNormalizedRGBPNG encodes normalized [-1,1] RGB pixels without a
// layout conversion. planar selects CHW; false selects HWC.
func EncodeNormalizedRGBPNG(pixels []float32, height, width int, planar bool) ([]byte, float32, float32, error) {
	if height <= 0 || width <= 0 || len(pixels) != height*width*RGBChannels {
		return nil, 0, 0, errors.New("media: invalid normalized RGB output")
	}
	minimum, maximum := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, value := range pixels {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, 0, errors.New("media: normalized RGB output is not finite")
		}
		minimum, maximum = min(minimum, value), max(maximum, value)
	}
	pixelStride, channelStride := RGBChannels, 1
	if planar {
		pixelStride, channelStride = 1, width*height
	}
	source := normalizedRGBImage{
		pixels: pixels, width: width, height: height,
		pixelStride: pixelStride, channelStride: channelStride,
	}
	var encoded bytes.Buffer
	encoder := png.Encoder{CompressionLevel: png.DefaultCompression}
	if err := encoder.Encode(&encoded, source); err != nil {
		return nil, 0, 0, err
	}
	return encoded.Bytes(), minimum, maximum, nil
}

// EncodePlanarRGB8Into converts finite planar RGB values to interleaved bytes
// using a caller-supplied quantizer.
func EncodePlanarRGB8Into(destination []byte, pixels []float32, height, width int, quantize func(float32) uint8) error {
	if quantize == nil || height <= 0 || width <= 0 || len(pixels) != height*width*RGBChannels || len(destination) != len(pixels) {
		return errors.New("media: invalid planar RGB byte conversion")
	}
	plane := height * width
	for index := range plane {
		for channel := range RGBChannels {
			value := pixels[channel*plane+index]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("media: planar RGB value is not finite")
			}
			destination[index*RGBChannels+channel] = quantize(value)
		}
	}
	return nil
}

func CloneRGBA(source *image.RGBA) *image.RGBA {
	result := image.NewRGBA(source.Bounds())
	copy(result.Pix, source.Pix)
	return result
}

func NormalizedRGBAChannel(value uint32) float32 {
	return float32(RGBAChannel8(value)) / float32(^uint8(0))
}

func RGBAChannel8(value uint32) uint8 { return uint8(value >> bitsPerByte) }

func RoundedUint8(value float64) uint8 {
	value = math.Floor(value + 0.5)
	if value <= 0 {
		return 0
	}
	if value >= float64(^uint8(0)) {
		return ^uint8(0)
	}
	return uint8(value)
}

func CubicConvolutionWeight(value float64) float64 {
	const coefficient = -0.75
	value = math.Abs(value)
	if value <= 1 {
		return ((coefficient+2)*value-(coefficient+3))*value*value + 1
	}
	if value < 2 {
		return ((coefficient*value-5*coefficient)*value+8*coefficient)*value - 4*coefficient
	}
	return 0
}

func ResizeBicubic(source image.Image, width, height int) *image.RGBA {
	bounds := source.Bounds()
	inputWidth, inputHeight := bounds.Dx(), bounds.Dy()
	xMin, xCount, xWeights := antialiasWeights(inputWidth, width)
	intermediate := make([][RGBChannels]uint8, inputHeight*width)
	for y := range inputHeight {
		for outX := range width {
			values := [RGBChannels]float64{}
			for offset := 0; offset < xCount[outX]; offset++ {
				r, g, b, _ := source.At(bounds.Min.X+xMin[outX]+offset, bounds.Min.Y+y).RGBA()
				weight := xWeights[outX][offset]
				values[0] += weight * float64(RGBAChannel8(r))
				values[1] += weight * float64(RGBAChannel8(g))
				values[2] += weight * float64(RGBAChannel8(b))
			}
			for channel := range values {
				intermediate[y*width+outX][channel] = RoundedUint8(values[channel])
			}
		}
	}
	yMin, yCount, yWeights := antialiasWeights(inputHeight, height)
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	for outY := range height {
		for x := range width {
			values := [RGBChannels]float64{}
			for offset := 0; offset < yCount[outY]; offset++ {
				pixel := intermediate[(yMin[outY]+offset)*width+x]
				weight := yWeights[outY][offset]
				for channel := range values {
					values[channel] += weight * float64(pixel[channel])
				}
			}
			index := output.PixOffset(x, outY)
			for channel := range values {
				output.Pix[index+channel] = RoundedUint8(values[channel])
			}
			output.Pix[index+RGBChannels] = ^uint8(0)
		}
	}
	return output
}

func antialiasWeights(input, output int) ([]int, []int, [][]float64) {
	scale := float64(input) / float64(output)
	support, inverseScale := 2.0, 1.0
	if scale >= 1 {
		support, inverseScale = 2*scale, 1/scale
	}
	minimum := make([]int, output)
	count := make([]int, output)
	weights := make([][]float64, output)
	for index := range output {
		center := scale * (float64(index) + RasterSampleCenter)
		low := max(0, int(center-support+RasterSampleCenter))
		high := min(input, int(center+support+RasterSampleCenter))
		values := make([]float64, high-low)
		total := 0.0
		for offset := range values {
			values[offset] = CubicConvolutionWeight((float64(offset+low) - center + RasterSampleCenter) * inverseScale)
			total += values[offset]
		}
		if total != 0 {
			for offset := range values {
				values[offset] /= total
			}
		}
		minimum[index], count[index], weights[index] = low, len(values), values
	}
	return minimum, count, weights
}

func InterleavedToPlanarRows(values []float32, rows, channels int) ([]float32, error) {
	if rows <= 0 || channels <= 0 || len(values)%rows != 0 {
		return nil, errors.New("media: invalid interleaved row storage")
	}
	rowWidth := len(values) / rows
	if rowWidth%channels != 0 {
		return nil, errors.New("media: interleaved row width is not channel-aligned")
	}
	pixels := rowWidth / channels
	output := make([]float32, len(values))
	for row := range rows {
		source, destination := values[row*rowWidth:], output[row*rowWidth:]
		for pixel := range pixels {
			for channel := range channels {
				destination[channel*pixels+pixel] = source[pixel*channels+channel]
			}
		}
	}
	return output, nil
}

func ResizeAligned(height, width, factor, minPixels, maxPixels int) (int, int, error) {
	if height <= 0 || width <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels {
		return 0, 0, errors.New("media: invalid aligned resize contract")
	}
	scale := float64(factor)
	resizedH := max(factor, int(math.RoundToEven(float64(height)/scale))*factor)
	resizedW := max(factor, int(math.RoundToEven(float64(width)/scale))*factor)
	switch {
	case resizedH*resizedW > maxPixels:
		beta := math.Sqrt(float64(height) * float64(width) / float64(maxPixels))
		resizedH = max(factor, int(math.Floor(float64(height)/beta/scale))*factor)
		resizedW = max(factor, int(math.Floor(float64(width)/beta/scale))*factor)
	case resizedH*resizedW < minPixels:
		beta := math.Sqrt(float64(minPixels) / (float64(height) * float64(width)))
		resizedH = int(math.Ceil(float64(height)*beta/scale)) * factor
		resizedW = int(math.Ceil(float64(width)*beta/scale)) * factor
	}
	return resizedH, resizedW, nil
}

func BestTileGrid(width, height, tile, minimum, maximum int) (int, int) {
	aspect := float64(width) / float64(height)
	bestW, bestH, bestDiff := 1, 1, math.Inf(1)
	for count := minimum; count <= maximum; count++ {
		for gridW := 1; gridW <= count; gridW++ {
			for gridH := 1; gridH <= count; gridH++ {
				tiles := gridW * gridH
				if tiles < minimum || tiles > maximum {
					continue
				}
				difference := math.Abs(aspect - float64(gridW)/float64(gridH))
				targetArea := tile * tile * tiles
				if difference < bestDiff || difference == bestDiff && width*height > targetArea/2 {
					bestW, bestH, bestDiff = gridW, gridH, difference
				}
			}
		}
	}
	return bestW, bestH
}
