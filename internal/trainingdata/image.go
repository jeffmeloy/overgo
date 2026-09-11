//overgo:runtime-inputs caller

package trainingdata

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"slices"
	"sort"

	"overgo/internal/binaryschema"
	"overgo/internal/recipecontract"
)

type structuredCrop struct {
	record RawRecord
	mad    float64
	ratio  float64
}

// SelectStructuredImageCrops ports adaptive_new's grid-corpus selector.
func SelectStructuredImageCrops(sources []RawRecord, size, samples int) ([]RawRecord, error) {
	if len(sources) == 0 || size < 2 || samples <= 0 {
		return nil, errors.New("training data: invalid structured image selection")
	}
	var candidates []structuredCrop
	for _, source := range sources {
		decoded, _, err := image.Decode(bytes.NewReader(source.Data))
		if err != nil {
			return nil, fmt.Errorf("training data: decode image %q: %w", source.ID, err)
		}
		bounds := decoded.Bounds()
		if bounds.Dx() < size || bounds.Dy() < size {
			return nil, fmt.Errorf("training data: image %q is smaller than crop %d", source.ID, size)
		}
		for tileY := 0; (tileY+1)*size <= bounds.Dy(); tileY++ {
			for tileX := 0; (tileX+1)*size <= bounds.Dx(); tileX++ {
				crop := image.NewNRGBA(image.Rect(0, 0, size, size))
				draw.Draw(crop, crop.Bounds(), decoded, image.Pt(bounds.Min.X+tileX*size, bounds.Min.Y+tileY*size), draw.Src)
				values := normalizedCHW(crop)
				var encoded bytes.Buffer
				if err := png.Encode(&encoded, crop); err != nil {
					return nil, err
				}
				id := fmt.Sprintf("%s/crop/%d/%d", source.ID, tileY, tileX)
				candidates = append(candidates, structuredCrop{
					record: RawRecord{ID: id, Group: source.Group, Data: encoded.Bytes()},
					mad:    medianAbsoluteDeviation(values), ratio: medianNeighbourRatio(values, 3, size),
				})
			}
		}
	}
	structured := candidates[:0]
	for _, candidate := range candidates {
		if candidate.ratio < 1 {
			structured = append(structured, candidate)
		}
	}
	if len(structured) < samples {
		return nil, fmt.Errorf("training data: only %d structured crops, want %d", len(structured), samples)
	}
	sort.SliceStable(structured, func(left, right int) bool { return structured[left].mad > structured[right].mad })
	population := structured[:len(structured)/2]
	if len(population) < samples {
		population = structured
	}
	selected := make([]RawRecord, samples)
	for index := range selected {
		selected[index] = population[index*len(population)/samples].record
	}
	return selected, nil
}

// ImageProcessor decodes one RGB image into normalized CHW values.
func ImageProcessor(role ValueRole) Processor {
	return func(_ context.Context, record RawRecord) (Example, error) {
		if !validRole(role) {
			return Example{}, errors.New("training data: invalid image role")
		}
		decoded, _, err := image.Decode(bytes.NewReader(record.Data))
		if err != nil {
			return Example{}, err
		}
		bounds := decoded.Bounds()
		width, height := bounds.Dx(), bounds.Dy()
		if width <= 0 || height <= 0 {
			return Example{}, errors.New("training data: empty image")
		}
		values := normalizedCHW(decoded)
		return Example{ID: record.ID, Group: record.Group, Values: []Value{{
			Role: role, Modality: recipecontract.ModalityImage, Encoding: EncodingFloat32LE,
			Shape: []int{3, height, width}, Data: binaryschema.LittleEndian.Float32s(values),
		}}}, nil
	}
}

// Image reconstructs one processor-owned normalized CHW image.
func Image(value Value) (image.Image, error) {
	if value.Modality != recipecontract.ModalityImage || len(value.Shape) != 3 || value.Shape[0] != 3 {
		return nil, errors.New("training data: invalid image value")
	}
	values, err := Float32(value)
	if err != nil {
		return nil, err
	}
	height, width := value.Shape[1], value.Shape[2]
	if len(values) != 3*height*width {
		return nil, errors.New("training data: image payload differs from shape")
	}
	result := image.NewNRGBA(image.Rect(0, 0, width, height))
	plane := height * width
	channel := func(input float32) uint8 {
		return uint8(math.Round(float64(min(max(input, -1), 1)+1) * 127.5))
	}
	for y := range height {
		for x := range width {
			index := y*width + x
			result.SetNRGBA(x, y, color.NRGBA{
				R: channel(values[index]), G: channel(values[plane+index]),
				B: channel(values[2*plane+index]), A: 255,
			})
		}
	}
	return result, nil
}

func normalizedCHW(source image.Image) []float32 {
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	values := make([]float32, 3*width*height)
	plane := width * height
	for y := range height {
		for x := range width {
			r, g, b, _ := source.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			index := y*width + x
			values[index] = float32(r)/32767.5 - 1
			values[plane+index] = float32(g)/32767.5 - 1
			values[2*plane+index] = float32(b)/32767.5 - 1
		}
	}
	return values
}

func medianAbsoluteDeviation(values []float32) float64 {
	center := medianFloat32(values)
	deviations := make([]float32, len(values))
	for index, value := range values {
		deviations[index] = float32(math.Abs(float64(value - center)))
	}
	return float64(medianFloat32(deviations))
}

func medianNeighbourRatio(values []float32, channels, size int) float64 {
	near := make([]float32, 0, channels*size*(size-1))
	far := make([]float32, 0, channels*size*(size-size/2))
	for channel := range channels {
		for y := range size {
			row := channel*size*size + y*size
			for x := 0; x+1 < size; x++ {
				near = append(near, float32(math.Abs(float64(values[row+x]-values[row+x+1]))))
			}
			for x := 0; x+size/2 < size; x++ {
				far = append(far, float32(math.Abs(float64(values[row+x]-values[row+x+size/2]))))
			}
		}
	}
	denominator := medianFloat32(far)
	if denominator == 0 {
		return 1
	}
	return float64(medianFloat32(near) / denominator)
}

func medianFloat32(values []float32) float32 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	middle := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[middle-1] + sorted[middle]) / 2
	}
	return sorted[middle]
}
