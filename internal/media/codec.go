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

// CodecCacheBehavior defines temporal state.
type CodecCacheBehavior uint8

const (
	// CodecCacheCausal retains causal prefix.
	CodecCacheCausal CodecCacheBehavior = iota + 1
	// CodecCacheReplicatedPrefix arms replicated-prefix transition.
	CodecCacheReplicatedPrefix
)

// CodecKernelABI identifies a bundled codec entry point.
type CodecKernelABI uint8

const (
	// CodecKernelConvolution runs causal convolution.
	CodecKernelConvolution CodecKernelABI = iota
	// CodecKernelRMSNorm runs channel RMS normalization.
	CodecKernelRMSNorm
	// CodecKernelAttention runs spatial attention.
	CodecKernelAttention
	// CodecKernelUpsample runs spatial upsample.
	CodecKernelUpsample
	// CodecKernelDownsample runs spatial downsample.
	CodecKernelDownsample
	// CodecKernelTemporalDownsample runs temporal downsample.
	CodecKernelTemporalDownsample
	// CodecKernelInterleave runs temporal interleave.
	CodecKernelInterleave
	// CodecKernelCacheUpdate runs temporal cache update.
	CodecKernelCacheUpdate
	// CodecKernelAdd runs residual addition.
	CodecKernelAdd
	// CodecKernelClamp clamps normalized output.
	CodecKernelClamp
	// CodecKernelABICount is the dense ABI extent.
	CodecKernelABICount
)

// CodecKernelFirstABI is the first dense ABI slot.
const CodecKernelFirstABI = CodecKernelConvolution

// CodecKernelSet records required ABIs.
type CodecKernelSet uint16

// Has reports ABI membership.
func (s CodecKernelSet) Has(kernel CodecKernelABI) bool { return s&(1<<kernel) != 0 }

// EntryPoint returns the bundled symbol.
func (k CodecKernelABI) EntryPoint() string {
	return [...]string{
		"vae_causal_conv3d_f32",
		"vae_channel_rms_norm_f32",
		"vae_spatial_attention_f32",
		"vae_upsample2d_f32",
		"vae_downsample2d_f32",
		"vae_temporal_downsample_f32",
		"vae_time_interleave_f32",
		"vae_temporal_cache_update_f32",
		"add_f32",
		"clamp_f32",
	}[k]
}

// CodecConvolutionExtents records compiled kernel and stride.
type CodecConvolutionExtents struct {
	Kernel [3]int
	Stride [3]int
}

// CodecWorkspaceLifetime identifies device allocation lifetime.
type CodecWorkspaceLifetime uint8

const (
	// CodecWorkspaceProgram spans operations.
	CodecWorkspaceProgram CodecWorkspaceLifetime = iota
	// CodecWorkspaceOperation belongs to one operation.
	CodecWorkspaceOperation
)

// CodecWorkspaceRole identifies allocation purpose.
type CodecWorkspaceRole uint8

const (
	// CodecWorkspaceActivation stores ping-pong activations.
	CodecWorkspaceActivation CodecWorkspaceRole = iota
	// CodecWorkspaceScratch stores reusable scratch.
	CodecWorkspaceScratch
	// CodecWorkspaceAttention stores attention scratch.
	CodecWorkspaceAttention
	// CodecWorkspaceCacheInput stores the first temporal cache.
	CodecWorkspaceCacheInput
	// CodecWorkspaceCacheOutput stores the second temporal cache.
	CodecWorkspaceCacheOutput
)

// CodecWorkspaceKey identifies a typed allocation.
type CodecWorkspaceKey struct {
	Lifetime  CodecWorkspaceLifetime
	Role      CodecWorkspaceRole
	Operation int
	Slot      int
}

// ProgramCodecWorkspace returns a program-lifetime key.
func ProgramCodecWorkspace(role CodecWorkspaceRole, slot int) CodecWorkspaceKey {
	return CodecWorkspaceKey{Lifetime: CodecWorkspaceProgram, Role: role, Slot: slot}
}

// OperationCodecWorkspace returns an operation-lifetime key.
func OperationCodecWorkspace(role CodecWorkspaceRole, operation, slot int) CodecWorkspaceKey {
	return CodecWorkspaceKey{Lifetime: CodecWorkspaceOperation, Role: role, Operation: operation, Slot: slot}
}

// CodecBindings stores operator parameters by role.
type CodecBindings[Binding any] struct {
	NormInput, WeightInput, BiasInput    Binding
	NormOutput, WeightOutput, BiasOutput Binding
	WeightProjection, BiasProjection     Binding
	WeightTemporal, BiasTemporal         Binding
	WeightSpatial, BiasSpatial           Binding
	bound                                bool
}

// BindCodecWeights validates and assigns ordered parameters.
func BindCodecWeights[Binding any](operator CodecOperator, projection bool, values []Binding) (CodecBindings[Binding], error) {
	want := len(CodecBindingValues(operator, projection, CodecBindings[Binding]{}))
	if len(values) != want {
		return CodecBindings[Binding]{}, fmt.Errorf("media: codec operator %d bindings=%d want=%d", operator, len(values), want)
	}
	b := CodecBindings[Binding]{bound: true}
	next := tensor.FirstOffset
	take := func() Binding {
		value := values[next]
		next++
		return value
	}
	switch operator {
	case CodecPointwise, CodecConvolution:
		b.WeightInput, b.BiasInput = take(), take()
	case CodecResidual:
		b.NormInput, b.WeightInput, b.BiasInput = take(), take(), take()
		b.NormOutput, b.WeightOutput, b.BiasOutput = take(), take(), take()
		if projection {
			b.WeightProjection, b.BiasProjection = take(), take()
		}
	case CodecAttention:
		b.NormInput, b.WeightInput, b.BiasInput = take(), take(), take()
		b.WeightOutput, b.BiasOutput = take(), take()
	case CodecDownsampleSpatial, CodecUpsampleSpatial:
		b.WeightSpatial, b.BiasSpatial = take(), take()
	case CodecDownsampleSpatiotemporal:
		b.WeightSpatial, b.BiasSpatial, b.WeightTemporal, b.BiasTemporal = take(), take(), take(), take()
	case CodecUpsampleSpatiotemporal:
		b.WeightTemporal, b.BiasTemporal, b.WeightSpatial, b.BiasSpatial = take(), take(), take(), take()
	case CodecHead:
		b.NormInput, b.WeightInput, b.BiasInput = take(), take(), take()
	}
	return b, nil
}

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
	Bindings                      CodecBindings[Bindings]
	Convolution                   CodecConvolutionExtents
	Cache                         CodecCacheBehavior
	Kernels                       CodecKernelSet
}

// NewCodecOperation compiles invariant execution facts.
func NewCodecOperation[Binding any](operator CodecOperator, name string, inputChannels, outputChannels int) CodecOperation[Binding] {
	unit := [3]int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent}
	triple := [3]int{tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent}
	op := CodecOperation[Binding]{Operator: operator, Name: name, InputChannels: inputChannels, OutputChannels: outputChannels, Convolution: CodecConvolutionExtents{Stride: unit}}
	set := func(kernels ...CodecKernelABI) {
		for _, kernel := range kernels {
			op.Kernels |= 1 << kernel
		}
	}
	switch operator {
	case CodecPointwise:
		op.Convolution.Kernel = unit
		set(CodecKernelConvolution)
	case CodecConvolution:
		op.Convolution.Kernel, op.Cache = triple, CodecCacheCausal
		set(CodecKernelConvolution, CodecKernelCacheUpdate)
	case CodecResidual:
		op.Convolution.Kernel, op.Cache = triple, CodecCacheCausal
		set(CodecKernelConvolution, CodecKernelRMSNorm, CodecKernelCacheUpdate, CodecKernelAdd)
	case CodecAttention:
		set(CodecKernelRMSNorm, CodecKernelAttention)
	case CodecDownsampleSpatial:
		op.Convolution.Kernel = [3]int{tensor.SingletonExtent, tensor.TripleExtent, tensor.TripleExtent}
		op.Convolution.Stride = [3]int{tensor.SingletonExtent, tensor.PairedExtent, tensor.PairedExtent}
		set(CodecKernelDownsample)
	case CodecDownsampleSpatiotemporal:
		op.Convolution.Kernel = triple
		op.Convolution.Stride = [3]int{tensor.PairedExtent, tensor.PairedExtent, tensor.PairedExtent}
		op.Cache = CodecCacheCausal
		set(CodecKernelDownsample, CodecKernelTemporalDownsample, CodecKernelCacheUpdate)
	case CodecUpsampleSpatial:
		op.Convolution.Kernel = [3]int{tensor.SingletonExtent, tensor.TripleExtent, tensor.TripleExtent}
		op.Convolution.Stride = [3]int{tensor.SingletonExtent, tensor.PairedExtent, tensor.PairedExtent}
		set(CodecKernelUpsample)
	case CodecUpsampleSpatiotemporal:
		op.Convolution.Kernel = triple
		op.Convolution.Stride = [3]int{tensor.PairedExtent, tensor.PairedExtent, tensor.PairedExtent}
		op.Cache = CodecCacheReplicatedPrefix
		set(CodecKernelConvolution, CodecKernelUpsample, CodecKernelInterleave, CodecKernelCacheUpdate)
	case CodecHead:
		op.Convolution.Kernel, op.Cache = triple, CodecCacheCausal
		set(CodecKernelRMSNorm, CodecKernelConvolution, CodecKernelCacheUpdate)
	}
	return op
}

// RequiresProjection reports whether a residual stage must project its input
// before addition.
func (o CodecOperation[Bindings]) RequiresProjection() bool {
	return o.Operator == CodecResidual && o.InputChannels != o.OutputChannels
}

// BindingValues returns artifact order.
func (o CodecOperation[Bindings]) BindingValues() []Bindings {
	return CodecBindingValues(o.Operator, o.RequiresProjection(), o.Bindings)
}

// CodecBindingValues returns artifact order for mapped storage.
func CodecBindingValues[Binding any](operator CodecOperator, projection bool, b CodecBindings[Binding]) []Binding {
	switch operator {
	case CodecPointwise, CodecConvolution:
		return []Binding{b.WeightInput, b.BiasInput}
	case CodecResidual:
		values := []Binding{b.NormInput, b.WeightInput, b.BiasInput, b.NormOutput, b.WeightOutput, b.BiasOutput}
		if projection {
			values = append(values, b.WeightProjection, b.BiasProjection)
		}
		return values
	case CodecAttention:
		return []Binding{b.NormInput, b.WeightInput, b.BiasInput, b.WeightOutput, b.BiasOutput}
	case CodecDownsampleSpatial, CodecUpsampleSpatial:
		return []Binding{b.WeightSpatial, b.BiasSpatial}
	case CodecDownsampleSpatiotemporal:
		return []Binding{b.WeightSpatial, b.BiasSpatial, b.WeightTemporal, b.BiasTemporal}
	case CodecUpsampleSpatiotemporal:
		return []Binding{b.WeightTemporal, b.BiasTemporal, b.WeightSpatial, b.BiasSpatial}
	case CodecHead:
		return []Binding{b.NormInput, b.WeightInput, b.BiasInput}
	}
	return nil
}

// CodecProgram: ordered validated codec topology.
type CodecProgram[Bindings any] struct {
	Operations []CodecOperation[Bindings]
}

// CodecVolume carries backend storage together with its current geometry.
type CodecVolume[Storage any] struct {
	Storage                         Storage
	Channels, Frames, Height, Width int
}

// CodecStep executes one compiled codec operation.
type CodecStep[Bindings, Storage, State any] func(
	int,
	CodecOperation[Bindings],
	*State,
	CodecVolume[Storage],
) (CodecVolume[Storage], error)

func codecOutputGeometry(operator CodecOperator, input VolumeGeometry) (VolumeGeometry, error) {
	scale := [3]int{operator.TemporalScale(), operator.SpatialScale(), operator.SpatialScale()}
	switch operator {
	case CodecDownsampleSpatial, CodecDownsampleSpatiotemporal:
		return DownsampledVolumeExactSpatial(input.Frames, input.Height, input.Width, scale)
	case CodecUpsampleSpatial, CodecUpsampleSpatiotemporal:
		return UpsampledCausalVolume(input.Frames, input.Height, input.Width, scale)
	default:
		return input, nil
	}
}

func (p CodecProgram[Bindings]) Validate(scope string) error {
	if len(p.Operations) == 0 {
		return fmt.Errorf("%s: operations absent", scope)
	}
	for index, operation := range p.Operations {
		expected := NewCodecOperation[Bindings](operation.Operator, operation.Name, operation.InputChannels, operation.OutputChannels)
		if operation.Operator == CodecOperatorNone || operation.Operator > CodecHead || operation.InputChannels <= 0 || operation.OutputChannels <= 0 || !operation.Bindings.bound || operation.Convolution != expected.Convolution || operation.Cache != expected.Cache || operation.Kernels != expected.Kernels {
			return fmt.Errorf("%s: operation %d (%s) is invalid", scope, index, operation.Name)
		}
		if index > 0 && operation.InputChannels != p.Operations[index-1].OutputChannels {
			return fmt.Errorf("%s: operation %d (%s) input channels=%d, previous output=%d",
				scope, index, operation.Name, operation.InputChannels, p.Operations[index-1].OutputChannels)
		}
	}
	return nil
}

// KernelABIs returns the program ABI union.
func (p CodecProgram[Bindings]) KernelABIs() CodecKernelSet {
	set := CodecKernelSet(1 << CodecKernelClamp)
	for _, operation := range p.Operations {
		set |= operation.Kernels
	}
	return set
}

// ExecuteCodecProgram owns stage order and geometry validation.
func ExecuteCodecProgram[Bindings, Storage, State any](
	scope string,
	program CodecProgram[Bindings],
	states []State,
	input CodecVolume[Storage],
	step CodecStep[Bindings, Storage, State],
) (CodecVolume[Storage], error) {
	if err := program.Validate(scope); err != nil {
		return input, err
	}
	if len(states) != len(program.Operations) || step == nil ||
		!checked.PositiveInts(input.Channels, input.Frames, input.Height, input.Width) ||
		!checked.Equal(input.Channels, program.Operations[tensor.FirstOffset].InputChannels) {
		return input, fmt.Errorf("%s: execution contract mismatch", scope)
	}
	current := input
	for index, operation := range program.Operations {
		if !checked.Equal(current.Channels, operation.InputChannels) {
			return current, fmt.Errorf("%s: operation %d (%s) input channels=%d, volume channels=%d", scope, index, operation.Name, operation.InputChannels, current.Channels)
		}
		geometry, geometryErr := codecOutputGeometry(operation.Operator, VolumeGeometry{
			Frames: current.Frames, Height: current.Height, Width: current.Width,
		})
		if geometryErr != nil {
			return current, fmt.Errorf("%s: operation %d (%s): %w", scope, index, operation.Name, geometryErr)
		}
		next, err := step(index, operation, &states[index], current)
		if err != nil {
			return current, fmt.Errorf("%s: operation %d (%s): %w", scope, index, operation.Name, err)
		}
		returned := VolumeGeometry{Frames: next.Frames, Height: next.Height, Width: next.Width}
		if !checked.Equal(next.Channels, operation.OutputChannels) ||
			returned != geometry {
			return current, fmt.Errorf("%s: operation %d (%s) returned invalid geometry", scope, index, operation.Name)
		}
		current = next
	}
	return current, nil
}

func (p CodecProgram[Bindings]) Names() []string {
	names := make([]string, len(p.Operations))
	for index, operation := range p.Operations {
		names[index] = operation.Name
	}
	return names
}
