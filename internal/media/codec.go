package media

import (
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

// VolumeGeometry is positive frame-height-width media geometry.
type VolumeGeometry struct {
	Frames, Height, Width int
}

// CausalChunkSchedule emits a cold-start singleton followed by stride-sized
// chunks, truncating only the final chunk.
func CausalChunkSchedule(frames, temporalStride int) []int {
	chunks := make([]int, tensor.FirstOffset, tensor.SingletonExtent+(frames-tensor.SingletonExtent+temporalStride-tensor.SingletonExtent)/temporalStride)
	for start := tensor.FirstOffset; start < frames; {
		count := tensor.SingletonExtent
		if checked.PositiveInts(start) {
			count = min(temporalStride, frames-start)
		}
		chunks = append(chunks, count)
		start += count
	}
	return chunks
}

// Convolution2DMatches compares a decoded two-dimensional convolution shape
// with the geometry required by a codec program.
func Convolution2DMatches(out, in, height, width, expectedOut, expectedIn, expectedHeight, expectedWidth int) bool {
	return out == expectedOut && in == expectedIn && height == expectedHeight && width == expectedWidth
}

// Convolution3DMatches compares a decoded three-dimensional convolution shape
// with the geometry required by a codec program.
func Convolution3DMatches(out, in, depth, height, width, expectedOut, expectedIn, expectedDepth, expectedHeight, expectedWidth int) bool {
	return out == expectedOut && in == expectedIn && depth == expectedDepth && height == expectedHeight && width == expectedWidth
}

// DownsampledVolume derives causal temporal and spatial codec geometry.
func DownsampledVolume(frames, height, width int, stride [3]int) (VolumeGeometry, error) {
	if !checked.PositiveInts(frames, height, width, stride[0], stride[1], stride[2]) {
		return VolumeGeometry{}, fmt.Errorf("media: invalid volume or codec stride")
	}
	return VolumeGeometry{
		Frames: (frames-1)/stride[0] + 1,
		Height: height / stride[1],
		Width:  width / stride[2],
	}, nil
}

// DownsampledVolumeExactSpatial derives causal temporal geometry while
// requiring exact spatial stride divisibility.
func DownsampledVolumeExactSpatial(frames, height, width int, stride [3]int) (VolumeGeometry, error) {
	volume, err := DownsampledVolume(frames, height, width, stride)
	if err != nil {
		return VolumeGeometry{}, err
	}
	if height%stride[1] != 0 || width%stride[2] != 0 {
		return VolumeGeometry{}, fmt.Errorf("media: spatial volume is not stride-aligned")
	}
	return volume, nil
}

// UpsampledCausalVolume derives the media geometry produced by a causal
// temporal codec and integral spatial upsampling.
func UpsampledCausalVolume(frames, height, width int, stride [3]int) (VolumeGeometry, error) {
	if !checked.PositiveInts(frames, height, width, stride[0], stride[1], stride[2]) {
		return VolumeGeometry{}, fmt.Errorf("media: invalid volume or codec stride")
	}
	temporal, ok := checked.MulInt(frames-1, stride[0])
	if !ok {
		return VolumeGeometry{}, fmt.Errorf("media: temporal volume overflows")
	}
	temporal, ok = checked.AddInt(temporal, 1)
	if !ok {
		return VolumeGeometry{}, fmt.Errorf("media: temporal volume overflows")
	}
	outputHeight, ok := checked.MulInt(height, stride[1])
	if !ok {
		return VolumeGeometry{}, fmt.Errorf("media: spatial volume overflows")
	}
	outputWidth, ok := checked.MulInt(width, stride[2])
	if !ok {
		return VolumeGeometry{}, fmt.Errorf("media: spatial volume overflows")
	}
	return VolumeGeometry{Frames: temporal, Height: outputHeight, Width: outputWidth}, nil
}

// VolumePatchGrid validates exact 3-D patch divisibility.
func VolumePatchGrid(volume VolumeGeometry, patch [3]int) ([3]int, error) {
	if !checked.PositiveInts(volume.Frames, volume.Height, volume.Width, patch[0], patch[1], patch[2]) ||
		volume.Frames%patch[0] != 0 || volume.Height%patch[1] != 0 || volume.Width%patch[2] != 0 {
		return [3]int{}, fmt.Errorf("media: volume %+v is incompatible with patch %v", volume, patch)
	}
	return [3]int{volume.Frames / patch[0], volume.Height / patch[1], volume.Width / patch[2]}, nil
}

// VolumeElements returns the overflow-checked element count of a 3-D extent.
func VolumeElements(extent [3]int) (int, error) {
	first, ok := checked.MulInt(extent[0], extent[1])
	if !ok {
		return 0, fmt.Errorf("media: volume extent overflows")
	}
	total, ok := checked.MulInt(first, extent[2])
	if !ok || total <= 0 {
		return 0, fmt.Errorf("media: volume extent is invalid")
	}
	return total, nil
}

// ResidualBindings appends projection bindings only when a residual stage
// changes channel width.
func ResidualBindings(bindings []string, inputChannels, outputChannels int, projection ...string) []string {
	result := slices.Clone(bindings)
	if inputChannels != outputChannels {
		result = append(result, projection...)
	}
	return result
}

const maximumDiagnosticBytes = 64 << 10

type DiagnosticBuffer struct{ data []byte }

func (w *DiagnosticBuffer) Write(data []byte) (int, error) {
	written := len(data)
	remaining := maximumDiagnosticBytes - len(w.data)
	if remaining > 0 {
		w.data = append(w.data, data[:min(len(data), remaining)]...)
	}
	return written, nil
}

func (w *DiagnosticBuffer) String() string { return string(w.data) }

// CodecOperator: typed media-codec stage.
type CodecOperator uint8

const (
	CodecOperatorNone CodecOperator = iota
	CodecPointwise
	CodecConvolution
	CodecResidual
	CodecAttention
	CodecDownsampleSpatial
	CodecDownsampleSpatiotemporal
	CodecUpsampleSpatial
	CodecUpsampleSpatiotemporal
	CodecHead
)

// SpatialScale returns the neutral spatial resampling factor encoded by the
// operator. Operators without spatial resampling preserve extent.
func (o CodecOperator) SpatialScale() int {
	switch o {
	case CodecDownsampleSpatial, CodecDownsampleSpatiotemporal,
		CodecUpsampleSpatial, CodecUpsampleSpatiotemporal:
		return tensor.PairedExtent
	default:
		return tensor.SingletonExtent
	}
}

// TemporalScale returns the neutral temporal resampling factor encoded by the
// operator. Operators without temporal resampling preserve extent.
func (o CodecOperator) TemporalScale() int {
	switch o {
	case CodecDownsampleSpatiotemporal, CodecUpsampleSpatiotemporal:
		return tensor.PairedExtent
	default:
		return tensor.SingletonExtent
	}
}

// ValidatePlanarGeometry validates positive channel-major image geometry.
func ValidatePlanarGeometry(channels, height, width int) error {
	if channels <= 0 {
		return fmt.Errorf("media: invalid planar geometry [%d,%d,%d]", channels, height, width)
	}
	return ValidateSpatialGeometry(height, width)
}

// ValidateSpatialGeometry validates positive two-dimensional media extent.
func ValidateSpatialGeometry(height, width int) error {
	if height <= 0 || width <= 0 {
		return fmt.Errorf("media: invalid spatial geometry %dx%d", width, height)
	}
	return nil
}

// DownsampledPlanarGeometry validates planar source geometry and derives the
// channel-major geometry produced by an integral spatial downsample.
func DownsampledPlanarGeometry(channels, height, width, scale int) (int, int, int, error) {
	if err := ValidatePlanarGeometry(channels, height, width); err != nil {
		return 0, 0, 0, err
	}
	if scale <= 0 {
		return 0, 0, 0, fmt.Errorf("media: invalid spatial scale %d", scale)
	}
	if height%scale != 0 || width%scale != 0 {
		return 0, 0, 0, fmt.Errorf("media: planar geometry %dx%d is not divisible by scale %d", height, width, scale)
	}
	return channels, height / scale, width / scale, nil
}

// CodecOperation: compiled stage plus runtime-owned bindings.
type CodecOperation[Bindings any] struct {
	Operator                      CodecOperator
	Name                          string
	InputChannels, OutputChannels int
	BindingCount                  int
	Bindings                      Bindings
}

// RequiresProjection reports whether a residual stage must project its input
// before addition.
func (o CodecOperation[Bindings]) RequiresProjection() bool {
	return o.Operator == CodecResidual && o.InputChannels != o.OutputChannels
}

// CodecProgram: ordered validated codec topology.
type CodecProgram[Bindings any] struct {
	Operations []CodecOperation[Bindings]
}

func (p CodecProgram[Bindings]) Validate(scope string) error {
	if len(p.Operations) == 0 {
		return fmt.Errorf("%s: operations absent", scope)
	}
	for index, operation := range p.Operations {
		if operation.Operator == CodecOperatorNone || operation.Operator > CodecHead || operation.BindingCount <= 0 || operation.InputChannels <= 0 || operation.OutputChannels <= 0 {
			return fmt.Errorf("%s: operation %d (%s) is invalid", scope, index, operation.Name)
		}
		if index > 0 && operation.InputChannels != p.Operations[index-1].OutputChannels {
			return fmt.Errorf("%s: operation %d (%s) input channels=%d, previous output=%d",
				scope, index, operation.Name, operation.InputChannels, p.Operations[index-1].OutputChannels)
		}
	}
	return nil
}

func (p CodecProgram[Bindings]) Names() []string {
	names := make([]string, len(p.Operations))
	for index, operation := range p.Operations {
		names[index] = operation.Name
	}
	return names
}
