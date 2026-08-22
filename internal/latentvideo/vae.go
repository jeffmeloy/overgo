// Wan 3-D causal VAE decoder: latent [zDim][F][h][w] -> RGB frames, streamed
// one latent frame (chunk) at a time exactly like the reference streamed
// plan (adaptive decodeCausalVideoCodecStreamedCUDAPlan, chunk-major).
// Structure discovers from checkpoint tensor names/shapes; the only external
// facts are the per-channel latent mean/std (published Wan stats, mirrored
// by the g0 capture — the checkpoint does not carry them). Temporal state is
// a per-convolution cache of the last <=2*padT input frames; the temporal
// upsample replays the reference 'Rep' first-chunk convention (skip the time
// conv on chunk 0, zero-prefixed cache on chunk 1). Ops accumulate in f64
// (host reference convention); output clamps to [-1,1] before emission.
package latentvideo

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

type vaePlanCore struct {
	media.CodecProgram[[]pytorchzip.TensorBinding]
	Stride               [tensor.TripleExtent]int
	UsedTensorCount      int
	UsedWeightBytes      int64
	LargestOpWeightBytes int64
}

// VAEDecoderPlan: validated decoder graph derived from tensor names/shapes.
type VAEDecoderPlan struct {
	vaePlanCore
	ZDim           int
	OutputChannels int
}

// VAELatentStats: per-channel latent normalization facts (decode applies
// z*std+mean before the first convolution, the reference upload affine).
type VAELatentStats struct {
	Mean, Std []float32
}

type vaeTensorMetadata struct {
	name       string
	dimensions []int64
}

type vaePlanStats struct {
	tensors        int
	weightBytes    int64
	largestOpBytes int64
}

type vaePlanCompiler struct {
	scope  string
	metas  []pytorchzip.TensorMeta
	byName map[string]pytorchzip.TensorMeta
	ops    []media.CodecOperation[[]pytorchzip.TensorBinding]
	names  []string
}

func newVAEPlanCompiler(scope string, metas []pytorchzip.TensorMeta) *vaePlanCompiler {
	compiler := &vaePlanCompiler{scope: scope, metas: metas, byName: make(map[string]pytorchzip.TensorMeta, len(metas))}
	for _, meta := range metas {
		compiler.byName[meta.Name] = meta
	}
	return compiler
}

func (c *vaePlanCompiler) tensorMetadata(name string) (vaeTensorMetadata, error) {
	meta, ok := c.byName[name]
	if !ok {
		return vaeTensorMetadata{}, fmt.Errorf("%s: missing tensor %s", c.scope, name)
	}
	return vaeTensorMetadata{name: name, dimensions: meta.Shape}, nil
}

func (c *vaePlanCompiler) vector(name string, elements int) error {
	meta, ok := c.byName[name]
	if !ok {
		return fmt.Errorf("%s: missing tensor %s", c.scope, name)
	}
	if meta.Numel != int64(elements) {
		return fmt.Errorf("%s: %s elements=%d want=%d", c.scope, name, meta.Numel, elements)
	}
	return nil
}

func (c *vaePlanCompiler) has(name string) bool {
	_, ok := c.byName[name]
	return ok
}

func (c *vaePlanCompiler) add(kind media.CodecOperator, prefix string, cIn, cOut int, names ...string) {
	c.ops = append(c.ops, media.CodecOperation[[]pytorchzip.TensorBinding]{
		Operator: kind, Name: prefix, InputChannels: cIn, OutputChannels: cOut, BindingCount: len(names),
	})
	c.names = append(c.names, names...)
}

func (c *vaePlanCompiler) finish(owned func(string) bool) (vaePlanStats, error) {
	consumed := make(map[string]bool, len(c.names))
	for _, name := range c.names {
		consumed[name] = true
	}
	var unexpected []string
	for _, meta := range c.metas {
		if owned(meta.Name) && !consumed[meta.Name] {
			unexpected = append(unexpected, meta.Name)
		}
	}
	if checked.Nonempty(unexpected) {
		sort.Strings(unexpected)
		return vaePlanStats{}, fmt.Errorf("%s: unconsumed tensors %v", c.scope, unexpected)
	}
	bindings, err := pytorchzip.CompileBindings(c.metas, c.names)
	if err != nil {
		return vaePlanStats{}, err
	}
	var stats vaePlanStats
	offset := tensor.FirstOffset
	for index := range c.ops {
		op := &c.ops[index]
		count := op.BindingCount
		op.Bindings = bindings[offset : offset+count]
		offset += count
		var operationBytes int64
		for _, binding := range op.Bindings {
			bytes, err := pytorchzip.TensorMetaBytes(binding.Meta)
			if err != nil {
				return vaePlanStats{}, err
			}
			operationBytes += bytes
		}
		stats.tensors += count
		stats.weightBytes += operationBytes
		stats.largestOpBytes = max(stats.largestOpBytes, operationBytes)
	}
	if !checked.Equal(offset, len(bindings)) {
		return vaePlanStats{}, fmt.Errorf("%s: binding partition consumed %d of %d", c.scope, offset, len(bindings))
	}
	return stats, nil
}

func (s vaeTensorMetadata) conv3d(label string) (cOut, cIn, kt, kh, kw int, err error) {
	if !checked.Equal(len(s.dimensions), tensor.MaxDimensions+tensor.SingletonExtent) {
		return 0, 0, 0, 0, 0, fmt.Errorf("vae decoder %s: %s shape %v is not rank-5", label, s.name, s.dimensions)
	}
	return int(s.dimensions[tensor.FirstOffset]), int(s.dimensions[tensor.SingletonExtent]), int(s.dimensions[tensor.PairedExtent]), int(s.dimensions[tensor.TripleExtent]), int(s.dimensions[tensor.MaxDimensions]), nil
}

func (s vaeTensorMetadata) conv2d(label string) (cOut, cIn, kh, kw int, err error) {
	if !checked.Equal(len(s.dimensions), tensor.MaxDimensions) {
		return 0, 0, 0, 0, fmt.Errorf("vae decoder %s: %s shape %v is not rank-4", label, s.name, s.dimensions)
	}
	return int(s.dimensions[tensor.FirstOffset]), int(s.dimensions[tensor.SingletonExtent]), int(s.dimensions[tensor.PairedExtent]), int(s.dimensions[tensor.TripleExtent]), nil
}

// CompileVAEDecoderPlan derives the decode graph from checkpoint metadata:
// conv2 (latent pointwise) -> decoder.conv1 -> decoder.middle.* ->
// decoder.upsamples.* -> decoder.head. Kinds resolve from which tensors
// exist under each prefix; channels resolve from weight shapes; the chain is
// validated end to end.
func CompileVAEDecoderPlan(metas []pytorchzip.TensorMeta) (VAEDecoderPlan, error) {
	compiler := newVAEPlanCompiler("vae decoder", metas)
	var plan VAEDecoderPlan

	// conv2: latent-space pointwise [z, z, 1, 1, 1].
	conv2, err := compiler.tensorMetadata("conv2.weight")
	if err != nil {
		return plan, err
	}
	zOut, zIn, kt, kh, kw, err := conv2.conv3d("conv2")
	if err != nil {
		return plan, err
	}
	if !media.Convolution3DMatches(zOut, zIn, kt, kh, kw, zIn, zIn, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
		return plan, fmt.Errorf("vae decoder: conv2.weight shape %v is not a latent pointwise", conv2.dimensions)
	}
	plan.ZDim = zIn
	compiler.add(media.CodecPointwise, "conv2", zIn, zOut, "conv2.weight", "conv2.bias")

	// decoder.conv1: latent -> feature volume conv [c, z, 3, 3, 3].
	conv1, err := compiler.tensorMetadata("decoder.conv1.weight")
	if err != nil {
		return plan, err
	}
	c1Out, c1In, kt, kh, kw, err := conv1.conv3d("conv1")
	if err != nil {
		return plan, err
	}
	if !media.Convolution3DMatches(c1Out, c1In, kt, kh, kw, c1Out, plan.ZDim, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
		return plan, fmt.Errorf("vae decoder: decoder.conv1.weight shape %v incompatible with z_dim=%d", conv1.dimensions, plan.ZDim)
	}
	compiler.add(media.CodecConvolution, "decoder.conv1", c1In, c1Out, "decoder.conv1.weight", "decoder.conv1.bias")
	channels := c1Out

	compileResidual := func(prefix string) (int, error) {
		w0, err := compiler.tensorMetadata(prefix + ".residual.2.weight")
		if err != nil {
			return 0, err
		}
		cOut, cIn, kt, kh, kw, err := w0.conv3d("residual")
		if err != nil {
			return 0, err
		}
		if !media.Convolution3DMatches(cOut, cIn, kt, kh, kw, cOut, channels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with input channels=%d", w0.name, w0.dimensions, channels)
		}
		w1, err := compiler.tensorMetadata(prefix + ".residual.6.weight")
		if err != nil {
			return 0, err
		}
		c1, c1In, kt, kh, kw, err := w1.conv3d("residual")
		if err != nil {
			return 0, err
		}
		if !media.Convolution3DMatches(c1, c1In, kt, kh, kw, cOut, cOut, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", w1.name, w1.dimensions, cOut)
		}
		for _, check := range []struct {
			name     string
			elements int
		}{
			{prefix + ".residual.0.gamma", cIn}, {prefix + ".residual.2.bias", cOut},
			{prefix + ".residual.3.gamma", cOut}, {prefix + ".residual.6.bias", cOut},
		} {
			if err := compiler.vector(check.name, check.elements); err != nil {
				return 0, err
			}
		}
		opNames := []string{
			prefix + ".residual.0.gamma", prefix + ".residual.2.weight", prefix + ".residual.2.bias",
			prefix + ".residual.3.gamma", prefix + ".residual.6.weight", prefix + ".residual.6.bias",
		}
		hasShortcut := compiler.has(prefix + ".shortcut.weight")
		if !checked.Equal(hasShortcut, !checked.Equal(cIn, cOut)) {
			return 0, fmt.Errorf("vae decoder: %s shortcut presence=%t but channels %d->%d", prefix, hasShortcut, cIn, cOut)
		}
		if hasShortcut {
			sw, err := compiler.tensorMetadata(prefix + ".shortcut.weight")
			if err != nil {
				return 0, err
			}
			sOut, sIn, kt, kh, kw, err := sw.conv3d("shortcut")
			if err != nil {
				return 0, err
			}
			if !media.Convolution3DMatches(sOut, sIn, kt, kh, kw, cOut, cIn, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
				return 0, fmt.Errorf("vae decoder: %s shape %v is not a %d->%d pointwise", sw.name, sw.dimensions, cIn, cOut)
			}
			if err := compiler.vector(prefix+".shortcut.bias", cOut); err != nil {
				return 0, err
			}
			opNames = append(opNames, prefix+".shortcut.weight", prefix+".shortcut.bias")
		}
		compiler.add(media.CodecResidual, prefix, cIn, cOut, opNames...)
		return cOut, nil
	}
	compileAttention := func(prefix string) error {
		qkvChannels, ok := checked.MulInt(tensor.TripleExtent, channels)
		if !ok {
			return fmt.Errorf("vae decoder: %s qkv channels overflow", prefix)
		}
		qkv, err := compiler.tensorMetadata(prefix + ".to_qkv.weight")
		if err != nil {
			return err
		}
		qkvOut, qkvIn, kh, kw, err := qkv.conv2d("attention")
		if err != nil {
			return err
		}
		if !media.Convolution2DMatches(qkvOut, qkvIn, kh, kw, qkvChannels, channels, tensor.SingletonExtent, tensor.SingletonExtent) {
			return fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", qkv.name, qkv.dimensions, channels)
		}
		proj, err := compiler.tensorMetadata(prefix + ".proj.weight")
		if err != nil {
			return err
		}
		projOut, projIn, kh, kw, err := proj.conv2d("attention")
		if err != nil {
			return err
		}
		if !media.Convolution2DMatches(projOut, projIn, kh, kw, channels, channels, tensor.SingletonExtent, tensor.SingletonExtent) {
			return fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", proj.name, proj.dimensions, channels)
		}
		for _, check := range []struct {
			name     string
			elements int
		}{
			{prefix + ".norm.gamma", channels}, {prefix + ".to_qkv.bias", qkvChannels}, {prefix + ".proj.bias", channels},
		} {
			if err := compiler.vector(check.name, check.elements); err != nil {
				return err
			}
		}
		compiler.add(media.CodecAttention, prefix, channels, channels,
			prefix+".norm.gamma", prefix+".to_qkv.weight", prefix+".to_qkv.bias",
			prefix+".proj.weight", prefix+".proj.bias")
		return nil
	}
	compileResample := func(prefix string) (int, error) {
		rw, err := compiler.tensorMetadata(prefix + ".resample.1.weight")
		if err != nil {
			return 0, err
		}
		cOut, cIn, _, _, err := rw.conv2d("resample")
		if err != nil {
			return 0, err
		}
		if !checked.Equal(cIn, channels) {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", rw.name, rw.dimensions, channels)
		}
		if err := compiler.vector(prefix+".resample.1.bias", cOut); err != nil {
			return 0, err
		}
		if !compiler.has(prefix + ".time_conv.weight") {
			operator := media.CodecUpsampleSpatial
			compiler.add(operator, prefix, cIn, cOut, prefix+".resample.1.weight", prefix+".resample.1.bias")
			plan.Stride[tensor.SingletonExtent] *= operator.SpatialScale()
			plan.Stride[tensor.PairedExtent] *= operator.SpatialScale()
			return cOut, nil
		}
		tw, err := compiler.tensorMetadata(prefix + ".time_conv.weight")
		if err != nil {
			return 0, err
		}
		tOut, tIn, kt, kh, kw, err := tw.conv3d("time_conv")
		if err != nil {
			return 0, err
		}
		temporalChannels, ok := checked.MulInt(tensor.PairedExtent, channels)
		if !ok || !media.Convolution3DMatches(tOut, tIn, kt, kh, kw, temporalChannels, channels, tensor.TripleExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
			return 0, fmt.Errorf("vae decoder: %s shape %v is not a %d->%d temporal conv", tw.name, tw.dimensions, channels, temporalChannels)
		}
		if err := compiler.vector(prefix+".time_conv.bias", tOut); err != nil {
			return 0, err
		}
		operator := media.CodecUpsampleSpatiotemporal
		compiler.add(operator, prefix, cIn, cOut,
			prefix+".time_conv.weight", prefix+".time_conv.bias",
			prefix+".resample.1.weight", prefix+".resample.1.bias")
		plan.Stride[tensor.FirstOffset] *= operator.TemporalScale()
		plan.Stride[tensor.SingletonExtent] *= operator.SpatialScale()
		plan.Stride[tensor.PairedExtent] *= operator.SpatialScale()
		return cOut, nil
	}

	plan.Stride = [tensor.TripleExtent]int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent}
	for _, group := range []string{"decoder.middle", "decoder.upsamples"} {
		count := tensor.FirstOffset
		for {
			prefix := fmt.Sprintf("%s.%d", group, count)
			isResidual := compiler.has(prefix + ".residual.0.gamma")
			isAttention := compiler.has(prefix + ".norm.gamma")
			isResample := compiler.has(prefix + ".resample.1.weight")
			switch {
			case isResidual:
				channels, err = compileResidual(prefix)
			case isAttention:
				err = compileAttention(prefix)
			case isResample:
				channels, err = compileResample(prefix)
			default:
				err = nil
			}
			if err != nil {
				return plan, err
			}
			if !isResidual && !isAttention && !isResample {
				break
			}
			count++
		}
		if checked.Equal(count, tensor.FirstOffset) {
			return plan, fmt.Errorf("vae decoder: no operations under %s", group)
		}
	}

	// decoder.head: norm+silu then conv to output channels.
	hw, err := compiler.tensorMetadata("decoder.head.2.weight")
	if err != nil {
		return plan, err
	}
	headOut, headIn, kt, kh, kw, err := hw.conv3d("head")
	if err != nil {
		return plan, err
	}
	if !media.Convolution3DMatches(headOut, headIn, kt, kh, kw, headOut, channels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
		return plan, fmt.Errorf("vae decoder: decoder.head.2.weight shape %v incompatible with channels=%d", hw.dimensions, channels)
	}
	if err := compiler.vector("decoder.head.0.gamma", channels); err != nil {
		return plan, err
	}
	if err := compiler.vector("decoder.head.2.bias", headOut); err != nil {
		return plan, err
	}
	compiler.add(media.CodecHead, "decoder.head", channels, headOut, "decoder.head.0.gamma", "decoder.head.2.weight", "decoder.head.2.bias")
	plan.OutputChannels = headOut
	plan.CodecProgram.Operations = compiler.ops

	if err := plan.CodecProgram.Validate("vae decoder"); err != nil {
		return plan, err
	}
	stats, err := compiler.finish(func(name string) bool {
		return strings.HasPrefix(name, "decoder.") || strings.HasPrefix(name, "conv2.")
	})
	if err != nil {
		return plan, err
	}
	plan.UsedTensorCount = stats.tensors
	plan.UsedWeightBytes = stats.weightBytes
	plan.LargestOpWeightBytes = stats.largestOpBytes
	return plan, nil
}

// VideoFrameSink consumes one borrowed planar [c][h][w] frame; sinks that
// retain it must copy.
type VideoFrameSink func(index int, frame []float32, height, width int) error

// VAEDecodeStats: measured streaming behavior; PeakHeapAllocBytes is the
// largest sampled Go heap live-set DELTA over the entry baseline (the
// EncoderStats convention).
type VAEDecodeStats struct {
	Ops                int
	LatentFrames       int
	OutputFrames       int
	OutputChannels     int
	OutputHeight       int
	OutputWidth        int
	WeightBytesRead    int64
	MaxOpWeightBytes   int64
	Engine             string
	DecodeWallSec      float64
	PeakHeapAllocBytes uint64
	// PeakDeviceBytes: runtime-owned device allocation peak (CUDA engine only).
	PeakDeviceBytes uint64
}

type vaeDecodeGeometry struct {
	spatial, frames, channels, height, width int
}

func prepareVAEDecode(scope, engine string, plan VAEDecoderPlan, opCount int, stats VAELatentStats, z []float32, latentFrames, latentH, latentW int, sink VideoFrameSink) (VAEDecodeStats, vaeDecodeGeometry, error) {
	var result VAEDecodeStats
	if sink == nil || !checked.PositiveInts(opCount) {
		return result, vaeDecodeGeometry{}, fmt.Errorf("%s: sink or operations absent", scope)
	}
	if err := media.ValidateChannelMoments(stats.Mean, stats.Std, plan.ZDim); err != nil {
		return result, vaeDecodeGeometry{}, fmt.Errorf("%s: latent stats: %w", scope, err)
	}
	spatial, ok := checked.MulInt(latentH, latentW)
	if !ok {
		return result, vaeDecodeGeometry{}, fmt.Errorf("%s: latent spatial extent overflows", scope)
	}
	if err := checked.Length(z, plan.ZDim, latentFrames, latentH, latentW); err != nil {
		return result, vaeDecodeGeometry{}, fmt.Errorf("%s: latent storage: %w", scope, err)
	}
	channels, frames, height, width, err := CompileVideoDecodeGeometry(plan, latentFrames, latentH, latentW)
	if err != nil {
		return result, vaeDecodeGeometry{}, err
	}
	result = VAEDecodeStats{
		Ops: opCount, LatentFrames: latentFrames, WeightBytesRead: plan.UsedWeightBytes,
		MaxOpWeightBytes: plan.LargestOpWeightBytes, OutputChannels: channels,
		OutputHeight: height, OutputWidth: width, Engine: engine,
	}
	return result, vaeDecodeGeometry{spatial: spatial, frames: frames, channels: channels, height: height, width: width}, nil
}

func denormalizeLatentChunk(dst, z []float32, stats VAELatentStats, channels, frames, spatial, chunk int) {
	for channel := range channels {
		source := z[(channel*frames+chunk)*spatial:]
		target := dst[channel*spatial:]
		mean, std := stats.Mean[channel], stats.Std[channel]
		for position := range spatial {
			target[position] = source[position]*std + mean
		}
	}
}

// vaeTemporalCache: per-convolution stream state — the last <=2 frames of
// that convolution's input (reference feat_cache). rep marks the temporal
// upsample's first-chunk placeholder (reference 'Rep': the next chunk's
// time conv runs cacheless and the stored cache is zero-prefixed).
type vaeTemporalCache struct {
	data        []float32
	frames      int
	initialized bool
	rep         bool
}

type vaeOpState struct {
	cache0, cache1 vaeTemporalCache
}

// nextTemporalCache: reference temporal_cache_update semantics (modes 0/1).
// mode 0: keep the last two input frames, joining the prior cache's final
// frame when the chunk is a single frame; mode 1 (replicatePrefix): zero
// frame then the chunk.
func nextTemporalCache(prior vaeTemporalCache, x []float32, c, frames, spatial int, replicatePrefix bool) vaeTemporalCache {
	nextFrames := min(tensor.PairedExtent, frames)
	if replicatePrefix || (!checked.AtLeastInt(frames, tensor.PairedExtent) && prior.initialized && checked.PositiveInts(prior.frames)) {
		nextFrames, _ = checked.AddInt(frames, tensor.SingletonExtent)
	}
	out := make([]float32, c*nextFrames*spatial)
	for ch := range c {
		for ti := range nextFrames {
			dst := out[(ch*nextFrames+ti)*spatial : (ch*nextFrames+ti+tensor.SingletonExtent)*spatial]
			switch {
			case replicatePrefix && checked.Equal(ti, tensor.FirstOffset):
				// zero prefix (already zeroed)
			case replicatePrefix:
				copy(dst, x[(ch*frames+ti-tensor.SingletonExtent)*spatial:])
			case checked.AtLeastInt(frames, tensor.PairedExtent):
				copy(dst, x[(ch*frames+frames-nextFrames+ti)*spatial:])
			case checked.PositiveInts(prior.frames) && checked.Equal(ti, tensor.FirstOffset):
				copy(dst, prior.data[(ch*prior.frames+checked.ReverseIndex(tensor.FirstOffset, prior.frames))*spatial:])
			default:
				src := ti
				if checked.PositiveInts(prior.frames) {
					src = ti - tensor.SingletonExtent
				}
				copy(dst, x[(ch*frames+src)*spatial:])
			}
		}
	}
	return vaeTemporalCache{data: out, frames: nextFrames, initialized: true}
}

// timeInterleaveInto: temporal upsample reshape — the time conv's doubled
// channel output [2c][t][s] interleaves to [c][2t][s], even output frames
// from the first channel half (reference reshape+stack).
func timeInterleaveInto(out, convolved []float32, c, frames, spatial int) error {
	want, ok := checked.ProductInt(tensor.PairedExtent, c, frames, spatial)
	if !ok || !checked.Equal(len(convolved), want) || !checked.Equal(len(out), len(convolved)) {
		return fmt.Errorf("vae time interleave: bad lengths out=%d convolved=%d", len(out), len(convolved))
	}
	outFrames, _ := checked.MulInt(tensor.PairedExtent, frames)
	for ch := range c {
		for ti := range outFrames {
			src := ((ti%tensor.PairedExtent)*c + ch) * frames * spatial
			copy(out[(ch*outFrames+ti)*spatial:(ch*outFrames+ti+tensor.SingletonExtent)*spatial],
				convolved[src+(ti/tensor.PairedExtent)*spatial:])
		}
	}
	return nil
}

// vaeLoadedOp: an op with weights resident (gammas flattened to channels).
type vaeLoadedOp struct {
	media.CodecOperation[[]pytorchzip.TensorBinding]
	values [][]float32
}

func loadVAEDecoderOps(reader *pytorchzip.Reader, plan VAEDecoderPlan) ([]vaeLoadedOp, error) {
	return loadVAEOps(reader, plan.vaePlanCore, "decoder")
}

func loadVAEOps(reader *pytorchzip.Reader, plan vaePlanCore, scope string) ([]vaeLoadedOp, error) {
	ops := make([]vaeLoadedOp, len(plan.Operations))
	for index, op := range plan.Operations {
		values, err := reader.ReadBindingValues(op.Bindings)
		if err != nil {
			return nil, fmt.Errorf("vae %s %s: %w", scope, op.Name, err)
		}
		ops[index] = vaeLoadedOp{CodecOperation: op, values: values}
	}
	return ops, nil
}

func causalGeometry(cIn, cOut, kt, kh, kw, padT, t, h, w int) hostmath.Conv3DShape {
	return hostmath.Conv3DShape{
		CIn: cIn, COut: cOut, InT: t, InH: h, InW: w,
		KT: kt, KH: kh, KW: kw,
		PadT: padT, PadH: kh / tensor.PairedExtent, PadW: kw / tensor.PairedExtent,
		StrideT: tensor.SingletonExtent, StrideH: tensor.SingletonExtent, StrideW: tensor.SingletonExtent,
	}
}

// runVAEOp executes one op on one chunk [cIn][frames][h][w], mutating the
// op's temporal state exactly as the reference chunk-major graph does.
func runVAEOp(op vaeLoadedOp, state *vaeOpState, chunkIndex int, x []float32, frames, h, w int) ([]float32, int, int, int, error) {
	spatial := h * w
	c := op.InputChannels
	cachedConv := func(out, input []float32, cache *vaeTemporalCache, weight, bias []float32, cOut, kt, kh, kw int) error {
		shape := causalGeometry(c, cOut, kt, kh, kw, kt/tensor.PairedExtent, frames, h, w)
		if err := hostmath.CausalConv3DInto(out, input, cache.data, weight, bias, cache.frames, shape); err != nil {
			return err
		}
		if checked.Nonzero(shape.PadT) {
			*cache = nextTemporalCache(*cache, input, c, frames, spatial, false)
		}
		return nil
	}
	switch op.Operator {
	case media.CodecPointwise, media.CodecConvolution:
		weight, bias := op.values[0], op.values[1]
		kt := tensor.SingletonExtent
		if op.Operator == media.CodecConvolution {
			kt = tensor.TripleExtent
		}
		out := make([]float32, op.OutputChannels*frames*spatial)
		if err := cachedConv(out, x, &state.cache0, weight, bias, op.OutputChannels, kt, kt, kt); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case media.CodecResidual:
		gamma0, w0, b0 := op.values[0], op.values[1], op.values[2]
		gamma1, w1, b1 := op.values[3], op.values[4], op.values[5]
		n0 := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(n0, x, gamma0, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n0)
		h0 := make([]float32, op.OutputChannels*frames*spatial)
		if err := cachedConv(h0, n0, &state.cache0, w0, b0, op.OutputChannels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent); err != nil {
			return nil, 0, 0, 0, err
		}
		n1 := make([]float32, len(h0))
		if err := hostmath.ChannelRMSNormF64Into(n1, h0, gamma1, op.OutputChannels, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n1)
		out := make([]float32, len(h0))
		shape1 := causalGeometry(op.OutputChannels, op.OutputChannels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent, tensor.SingletonExtent, frames, h, w)
		if err := hostmath.CausalConv3DInto(out, n1, state.cache1.data, w1, b1, state.cache1.frames, shape1); err != nil {
			return nil, 0, 0, 0, err
		}
		state.cache1 = nextTemporalCache(state.cache1, n1, op.OutputChannels, frames, spatial, false)
		if !op.RequiresProjection() {
			for i := range out {
				out[i] += x[i]
			}
			return out, frames, h, w, nil
		}
		shortcut := make([]float32, len(out))
		projection, ok := checked.Suffix(op.values, tensor.PairedExtent)
		if !ok {
			return nil, 0, 0, 0, fmt.Errorf("vae residual projection bindings are absent")
		}
		if err := hostmath.ChannelMixF64Into(shortcut, x, projection[tensor.FirstOffset], projection[tensor.SingletonExtent], c, op.OutputChannels, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += shortcut[i]
		}
		return out, frames, h, w, nil
	case media.CodecAttention:
		gamma, qkvW, qkvB := op.values[0], op.values[1], op.values[2]
		projW, projB := op.values[3], op.values[4]
		norm := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(norm, x, gamma, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		qkvChannels, ok := checked.MulInt(tensor.TripleExtent, c)
		qkvElements, elementsOK := checked.ProductInt(qkvChannels, frames, spatial)
		if !ok || !elementsOK {
			return nil, 0, 0, 0, fmt.Errorf("vae attention geometry overflows")
		}
		qkv := make([]float32, qkvElements)
		if err := hostmath.ChannelMixF64Into(qkv, norm, qkvW, qkvB, c, qkvChannels, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		if err := hostmath.SpatialAttentionF64Into(norm, qkv, c, frames, spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		out := make([]float32, len(x))
		if err := hostmath.ChannelMixF64Into(out, norm, projW, projB, c, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += x[i]
		}
		return out, frames, h, w, nil
	case media.CodecDownsampleSpatial:
		scale := op.Operator.SpatialScale()
		out := make([]float32, op.OutputChannels*frames*(h/scale)*(w/scale))
		if err := hostmath.Downsample2DChannelsF64Into(out, x, op.values[0], op.values[1], c, frames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, h / scale, w / scale, nil
	case media.CodecDownsampleSpatiotemporal:
		scale := op.Operator.SpatialScale()
		outH, outW := h/scale, w/scale
		spatialOut := make([]float32, op.OutputChannels*frames*outH*outW)
		if err := hostmath.Downsample2DChannelsF64Into(spatialOut, x, op.values[0], op.values[1], c, frames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		prior := state.cache0
		state.cache0 = nextTemporalCache(state.cache0, spatialOut, op.OutputChannels, frames, outH*outW, false)
		if checked.Equal(chunkIndex, tensor.FirstOffset) {
			return spatialOut, frames, outH, outW, nil
		}
		joinedFrames, _ := checked.AddInt(frames, tensor.SingletonExtent)
		joined := make([]float32, op.OutputChannels*joinedFrames*outH*outW)
		for channel := range op.OutputChannels {
			if checked.PositiveInts(prior.frames) {
				destination := joined[channel*joinedFrames*outH*outW : (channel*joinedFrames+tensor.SingletonExtent)*outH*outW]
				source := prior.data[(channel*prior.frames+checked.ReverseIndex(tensor.FirstOffset, prior.frames))*outH*outW : (channel*prior.frames+prior.frames)*outH*outW]
				copy(destination, source)
			}
			destination := joined[(channel*joinedFrames+tensor.SingletonExtent)*outH*outW : (channel+tensor.SingletonExtent)*joinedFrames*outH*outW]
			source := spatialOut[channel*frames*outH*outW : (channel+tensor.SingletonExtent)*frames*outH*outW]
			copy(destination, source)
		}
		shape := hostmath.Conv3DShape{
			CIn: op.OutputChannels, COut: op.OutputChannels, InT: joinedFrames, InH: outH, InW: outW,
			KT: tensor.TripleExtent, KH: tensor.SingletonExtent, KW: tensor.SingletonExtent,
			StrideT: op.Operator.TemporalScale(), StrideH: tensor.SingletonExtent, StrideW: tensor.SingletonExtent,
		}
		outFrames, _, _, err := shape.OutputDims()
		if err != nil {
			return nil, 0, 0, 0, err
		}
		out := make([]float32, op.OutputChannels*outFrames*outH*outW)
		if err := hostmath.CausalConv3DInto(out, joined, nil, op.values[tensor.PairedExtent], op.values[tensor.TripleExtent], tensor.FirstOffset, shape); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, outFrames, outH, outW, nil
	case media.CodecUpsampleSpatial:
		weight, bias := op.values[0], op.values[1]
		scale := op.Operator.SpatialScale()
		out := make([]float32, op.OutputChannels*frames*scale*h*scale*w)
		if err := hostmath.ResizeConv2DInto(out, x, weight, bias, c, op.OutputChannels, frames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, scale * h, scale * w, nil
	case media.CodecUpsampleSpatiotemporal:
		timeW, timeB := op.values[0], op.values[1]
		resampleW, resampleB := op.values[2], op.values[3]
		spatialInput, spatialFrames := x, frames
		if checked.Equal(chunkIndex, tensor.FirstOffset) {
			state.cache0 = vaeTemporalCache{initialized: true, rep: true}
		} else {
			replicatePrefix := state.cache0.rep && !checked.AtLeastInt(frames, tensor.PairedExtent)
			nextCache := nextTemporalCache(state.cache0, x, c, frames, spatial, replicatePrefix)
			cacheData, cacheFrames := state.cache0.data, state.cache0.frames
			if state.cache0.rep {
				cacheData, cacheFrames = nil, tensor.FirstOffset
			}
			temporalChannels, ok := checked.MulInt(tensor.PairedExtent, c)
			timeElements, elementsOK := checked.ProductInt(temporalChannels, frames, spatial)
			if !ok || !elementsOK {
				return nil, 0, 0, 0, fmt.Errorf("vae temporal upsample geometry overflows")
			}
			timeShape := causalGeometry(c, temporalChannels, tensor.TripleExtent, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent, frames, h, w)
			timeOut := make([]float32, timeElements)
			if err := hostmath.CausalConv3DInto(timeOut, x, cacheData, timeW, timeB, cacheFrames, timeShape); err != nil {
				return nil, 0, 0, 0, err
			}
			spatialFrames, _ = checked.MulInt(tensor.PairedExtent, frames)
			spatialInput = make([]float32, c*spatialFrames*spatial)
			if err := timeInterleaveInto(spatialInput, timeOut, c, frames, spatial); err != nil {
				return nil, 0, 0, 0, err
			}
			state.cache0 = nextCache
		}
		scale := op.Operator.SpatialScale()
		out := make([]float32, op.OutputChannels*spatialFrames*scale*h*scale*w)
		if err := hostmath.ResizeConv2DInto(out, spatialInput, resampleW, resampleB, c, op.OutputChannels, spatialFrames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, spatialFrames, scale * h, scale * w, nil
	case media.CodecHead:
		gamma, weight, bias := op.values[0], op.values[1], op.values[2]
		norm := make([]float32, len(x))
		if err := hostmath.ChannelRMSNormF64Into(norm, x, gamma, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(norm)
		out := make([]float32, op.OutputChannels*frames*spatial)
		if err := cachedConv(out, norm, &state.cache0, weight, bias, op.OutputChannels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	}
	return nil, 0, 0, 0, fmt.Errorf("vae decoder: unsupported op %d", op.Operator)
}

// CompileVideoDecodeGeometry returns output extents for a latent volume under the plan's
// derived strides (frames = (F-1)*strideT + 1; the first latent frame decodes
// to a single frame, every later one to strideT frames).
func CompileVideoDecodeGeometry(plan VAEDecoderPlan, latentFrames, latentH, latentW int) (channels, frames, height, width int, err error) {
	if !checked.PositiveInts(plan.ZDim, plan.OutputChannels) {
		return 0, 0, 0, 0, fmt.Errorf("vae decode shape: invalid plan %+v", plan.Stride)
	}
	volume, err := media.UpsampledCausalVolume(latentFrames, latentH, latentW, plan.Stride)
	if err != nil {
		return 0, 0, 0, 0, fmt.Errorf("vae decode shape: %w", err)
	}
	return plan.OutputChannels, volume.Frames, volume.Height, volume.Width, nil
}

// DecodeLatentVideo streams the decode one latent frame at a time: denorm
// (z*std+mean), the compiled op chain with per-op temporal caches, clamp to
// [-1,1], then per-frame emission through the sink. Weights load once and
// stay resident (the reference chunk-major graph residency); peak host
// memory is weights plus one chunk's activations.
func DecodeLatentVideo(checkpoint string, plan VAEDecoderPlan, stats VAELatentStats, z []float32, latentFrames, latentH, latentW int, sink VideoFrameSink) (VAEDecodeStats, error) {
	decodeStats, geometry, err := prepareVAEDecode("vae decode", "host_streamed_chunks", plan, len(plan.Operations), stats, z, latentFrames, latentH, latentW, sink)
	if err != nil {
		return decodeStats, err
	}
	spatial := geometry.spatial
	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)
	samplePeak := func() {
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		if ms.HeapAlloc > baseline.HeapAlloc && ms.HeapAlloc-baseline.HeapAlloc > decodeStats.PeakHeapAllocBytes {
			decodeStats.PeakHeapAllocBytes = ms.HeapAlloc - baseline.HeapAlloc
		}
	}
	started := time.Now()
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return decodeStats, err
	}
	defer reader.Close()
	ops, err := loadVAEDecoderOps(reader, plan)
	if err != nil {
		return decodeStats, err
	}
	samplePeak()
	states := make([]vaeOpState, len(ops))
	frameScratch := make([]float32, 0)
	frameIndex := tensor.FirstOffset
	for chunkIndex := range latentFrames {
		x := make([]float32, plan.ZDim*spatial)
		denormalizeLatentChunk(x, z, stats, plan.ZDim, latentFrames, spatial, chunkIndex)
		frames, h, w := tensor.SingletonExtent, latentH, latentW
		for opIndex := range ops {
			x, frames, h, w, err = runVAEOp(ops[opIndex], &states[opIndex], chunkIndex, x, frames, h, w)
			if err != nil {
				return decodeStats, fmt.Errorf("vae decode %s chunk %d: %w", ops[opIndex].Name, chunkIndex, err)
			}
			samplePeak()
		}
		if !checked.Equal(h, geometry.height) || !checked.Equal(w, geometry.width) {
			return decodeStats, fmt.Errorf("vae decode chunk %d output %dx%d, want %dx%d", chunkIndex, w, h, geometry.width, geometry.height)
		}
		media.ClampNormalizedF32InPlace(x)
		chunkSpatial, ok := checked.MulInt(h, w)
		if !ok {
			return decodeStats, fmt.Errorf("vae decode chunk %d spatial geometry overflows", chunkIndex)
		}
		frameElements, ok := checked.MulInt(geometry.channels, chunkSpatial)
		if !ok {
			return decodeStats, fmt.Errorf("vae decode chunk %d frame geometry overflows", chunkIndex)
		}
		if cap(frameScratch) < frameElements {
			frameScratch = make([]float32, frameElements)
		}
		frame := frameScratch[:frameElements]
		for chunkFrame := range frames {
			for ch := range geometry.channels {
				copy(frame[ch*chunkSpatial:(ch+tensor.SingletonExtent)*chunkSpatial], x[(ch*frames+chunkFrame)*chunkSpatial:])
			}
			if err := sink(frameIndex, frame, h, w); err != nil {
				return decodeStats, fmt.Errorf("vae decode frame sink %d: %w", frameIndex, err)
			}
			frameIndex++
		}
		samplePeak()
	}
	if !checked.Equal(frameIndex, geometry.frames) {
		return decodeStats, fmt.Errorf("vae decode produced %d frames, want %d", frameIndex, geometry.frames)
	}
	decodeStats.OutputFrames = frameIndex
	decodeStats.DecodeWallSec = time.Since(started).Seconds()
	return decodeStats, nil
}
