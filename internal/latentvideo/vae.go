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
	"math"
	"runtime"
	"sort"
	"strings"
	"time"

	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
)

// channelNormZeroGuard: zero-column RMS floor (reference constant).
const channelNormZeroGuard = 1e-12

// vaeSpatialScale: every resample stage is a fixed 2x (Wan resample stack).
const vaeSpatialScale = 2

type vaeOpKind string

const (
	vaeOpPointwise    vaeOpKind = "pointwise3d"
	vaeOpConv         vaeOpKind = "conv3d"
	vaeOpResidual     vaeOpKind = "residual"
	vaeOpAttention    vaeOpKind = "attention"
	vaeOpDownsample2D vaeOpKind = "downsample2d"
	vaeOpDownsample3D vaeOpKind = "downsample3d"
	vaeOpUpsample2D   vaeOpKind = "upsample2d"
	vaeOpUpsample3D   vaeOpKind = "upsample3d"
	vaeOpHead         vaeOpKind = "head"
)

// vaeDecoderOp: one compiled decode operation with tensor bindings in
// canonical order (gammas and weight/bias pairs as each op consumes them).
type vaeDecoderOp struct {
	kind        vaeOpKind
	prefix      string
	cIn, cOut   int
	bindings    []pytorchzip.TensorBinding
	weightBytes int64
}

type vaePlanCore struct {
	Stride               [3]int
	UsedTensorCount      int
	UsedWeightBytes      int64
	LargestOpWeightBytes int64
	ops                  []vaeDecoderOp
}

func (p vaePlanCore) Ops() int { return len(p.ops) }

func (p vaePlanCore) OpPrefixes() []string {
	prefixes := make([]string, len(p.ops))
	for index, op := range p.ops {
		prefixes[index] = op.prefix
	}
	return prefixes
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

type vaeTensorShape struct {
	name  string
	shape []int64
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
	ops    []vaeDecoderOp
	names  []string
}

func newVAEPlanCompiler(scope string, metas []pytorchzip.TensorMeta) *vaePlanCompiler {
	compiler := &vaePlanCompiler{scope: scope, metas: metas, byName: make(map[string]pytorchzip.TensorMeta, len(metas))}
	for _, meta := range metas {
		compiler.byName[meta.Name] = meta
	}
	return compiler
}

func (c *vaePlanCompiler) shape(name string) (vaeTensorShape, error) {
	meta, ok := c.byName[name]
	if !ok {
		return vaeTensorShape{}, fmt.Errorf("%s: missing tensor %s", c.scope, name)
	}
	return vaeTensorShape{name: name, shape: meta.Shape}, nil
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

func (c *vaePlanCompiler) add(kind vaeOpKind, prefix string, cIn, cOut int, names ...string) {
	c.ops = append(c.ops, vaeDecoderOp{kind: kind, prefix: prefix, cIn: cIn, cOut: cOut})
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
	if len(unexpected) > 0 {
		sort.Strings(unexpected)
		return vaePlanStats{}, fmt.Errorf("%s: unconsumed tensors %v", c.scope, unexpected)
	}
	bindings, err := pytorchzip.CompileBindings(c.metas, c.names)
	if err != nil {
		return vaePlanStats{}, err
	}
	var stats vaePlanStats
	offset := 0
	for index := range c.ops {
		op := &c.ops[index]
		count := vaeOpTensorCount(*op)
		op.bindings = bindings[offset : offset+count]
		offset += count
		for _, binding := range op.bindings {
			bytes, err := pytorchzip.TensorMetaBytes(binding.Meta)
			if err != nil {
				return vaePlanStats{}, err
			}
			op.weightBytes += bytes
		}
		stats.tensors += count
		stats.weightBytes += op.weightBytes
		stats.largestOpBytes = max(stats.largestOpBytes, op.weightBytes)
	}
	if offset != len(bindings) {
		return vaePlanStats{}, fmt.Errorf("%s: binding partition consumed %d of %d", c.scope, offset, len(bindings))
	}
	return stats, nil
}

func (s vaeTensorShape) conv3d(label string) (cOut, cIn, kt, kh, kw int, err error) {
	if len(s.shape) != 5 {
		return 0, 0, 0, 0, 0, fmt.Errorf("vae decoder %s: %s shape %v is not rank-5", label, s.name, s.shape)
	}
	return int(s.shape[0]), int(s.shape[1]), int(s.shape[2]), int(s.shape[3]), int(s.shape[4]), nil
}

func (s vaeTensorShape) conv2d(label string) (cOut, cIn, kh, kw int, err error) {
	if len(s.shape) != 4 {
		return 0, 0, 0, 0, fmt.Errorf("vae decoder %s: %s shape %v is not rank-4", label, s.name, s.shape)
	}
	return int(s.shape[0]), int(s.shape[1]), int(s.shape[2]), int(s.shape[3]), nil
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
	conv2, err := compiler.shape("conv2.weight")
	if err != nil {
		return plan, err
	}
	zOut, zIn, kt, kh, kw, err := conv2.conv3d("conv2")
	if err != nil {
		return plan, err
	}
	if zOut != zIn || kt != 1 || kh != 1 || kw != 1 {
		return plan, fmt.Errorf("vae decoder: conv2.weight shape %v is not a latent pointwise", conv2.shape)
	}
	plan.ZDim = zIn
	compiler.add(vaeOpPointwise, "conv2", zIn, zOut, "conv2.weight", "conv2.bias")

	// decoder.conv1: latent -> feature volume conv [c, z, 3, 3, 3].
	conv1, err := compiler.shape("decoder.conv1.weight")
	if err != nil {
		return plan, err
	}
	c1Out, c1In, kt, kh, kw, err := conv1.conv3d("conv1")
	if err != nil {
		return plan, err
	}
	if c1In != plan.ZDim || kt != 3 || kh != 3 || kw != 3 {
		return plan, fmt.Errorf("vae decoder: decoder.conv1.weight shape %v incompatible with z_dim=%d", conv1.shape, plan.ZDim)
	}
	compiler.add(vaeOpConv, "decoder.conv1", c1In, c1Out, "decoder.conv1.weight", "decoder.conv1.bias")
	channels := c1Out

	compileResidual := func(prefix string) (int, error) {
		w0, err := compiler.shape(prefix + ".residual.2.weight")
		if err != nil {
			return 0, err
		}
		cOut, cIn, kt, kh, kw, err := w0.conv3d("residual")
		if err != nil {
			return 0, err
		}
		if cIn != channels || kt != 3 || kh != 3 || kw != 3 {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with input channels=%d", w0.name, w0.shape, channels)
		}
		w1, err := compiler.shape(prefix + ".residual.6.weight")
		if err != nil {
			return 0, err
		}
		c1, c1In, kt, kh, kw, err := w1.conv3d("residual")
		if err != nil {
			return 0, err
		}
		if c1 != cOut || c1In != cOut || kt != 3 || kh != 3 || kw != 3 {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", w1.name, w1.shape, cOut)
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
		if hasShortcut != (cIn != cOut) {
			return 0, fmt.Errorf("vae decoder: %s shortcut presence=%t but channels %d->%d", prefix, hasShortcut, cIn, cOut)
		}
		if hasShortcut {
			sw, err := compiler.shape(prefix + ".shortcut.weight")
			if err != nil {
				return 0, err
			}
			sOut, sIn, kt, kh, kw, err := sw.conv3d("shortcut")
			if err != nil {
				return 0, err
			}
			if sOut != cOut || sIn != cIn || kt != 1 || kh != 1 || kw != 1 {
				return 0, fmt.Errorf("vae decoder: %s shape %v is not a %d->%d pointwise", sw.name, sw.shape, cIn, cOut)
			}
			if err := compiler.vector(prefix+".shortcut.bias", cOut); err != nil {
				return 0, err
			}
			opNames = append(opNames, prefix+".shortcut.weight", prefix+".shortcut.bias")
		}
		compiler.add(vaeOpResidual, prefix, cIn, cOut, opNames...)
		return cOut, nil
	}
	compileAttention := func(prefix string) error {
		qkv, err := compiler.shape(prefix + ".to_qkv.weight")
		if err != nil {
			return err
		}
		qkvOut, qkvIn, kh, kw, err := qkv.conv2d("attention")
		if err != nil {
			return err
		}
		if qkvIn != channels || qkvOut != 3*channels || kh != 1 || kw != 1 {
			return fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", qkv.name, qkv.shape, channels)
		}
		proj, err := compiler.shape(prefix + ".proj.weight")
		if err != nil {
			return err
		}
		projOut, projIn, kh, kw, err := proj.conv2d("attention")
		if err != nil {
			return err
		}
		if projOut != channels || projIn != channels || kh != 1 || kw != 1 {
			return fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", proj.name, proj.shape, channels)
		}
		for _, check := range []struct {
			name     string
			elements int
		}{
			{prefix + ".norm.gamma", channels}, {prefix + ".to_qkv.bias", 3 * channels}, {prefix + ".proj.bias", channels},
		} {
			if err := compiler.vector(check.name, check.elements); err != nil {
				return err
			}
		}
		compiler.add(vaeOpAttention, prefix, channels, channels,
			prefix+".norm.gamma", prefix+".to_qkv.weight", prefix+".to_qkv.bias",
			prefix+".proj.weight", prefix+".proj.bias")
		return nil
	}
	compileResample := func(prefix string) (int, error) {
		rw, err := compiler.shape(prefix + ".resample.1.weight")
		if err != nil {
			return 0, err
		}
		cOut, cIn, _, _, err := rw.conv2d("resample")
		if err != nil {
			return 0, err
		}
		if cIn != channels {
			return 0, fmt.Errorf("vae decoder: %s shape %v incompatible with channels=%d", rw.name, rw.shape, channels)
		}
		if err := compiler.vector(prefix+".resample.1.bias", cOut); err != nil {
			return 0, err
		}
		if !compiler.has(prefix + ".time_conv.weight") {
			compiler.add(vaeOpUpsample2D, prefix, cIn, cOut, prefix+".resample.1.weight", prefix+".resample.1.bias")
			plan.Stride[1] *= vaeSpatialScale
			plan.Stride[2] *= vaeSpatialScale
			return cOut, nil
		}
		tw, err := compiler.shape(prefix + ".time_conv.weight")
		if err != nil {
			return 0, err
		}
		tOut, tIn, kt, kh, kw, err := tw.conv3d("time_conv")
		if err != nil {
			return 0, err
		}
		if tIn != channels || tOut != 2*channels || kt != 3 || kh != 1 || kw != 1 {
			return 0, fmt.Errorf("vae decoder: %s shape %v is not a %d->%d temporal conv", tw.name, tw.shape, channels, 2*channels)
		}
		if err := compiler.vector(prefix+".time_conv.bias", tOut); err != nil {
			return 0, err
		}
		compiler.add(vaeOpUpsample3D, prefix, cIn, cOut,
			prefix+".time_conv.weight", prefix+".time_conv.bias",
			prefix+".resample.1.weight", prefix+".resample.1.bias")
		plan.Stride[0] *= vaeSpatialScale
		plan.Stride[1] *= vaeSpatialScale
		plan.Stride[2] *= vaeSpatialScale
		return cOut, nil
	}

	plan.Stride = [3]int{1, 1, 1}
	for _, group := range []string{"decoder.middle", "decoder.upsamples"} {
		count := 0
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
		if count == 0 {
			return plan, fmt.Errorf("vae decoder: no operations under %s", group)
		}
	}

	// decoder.head: norm+silu then conv to output channels.
	hw, err := compiler.shape("decoder.head.2.weight")
	if err != nil {
		return plan, err
	}
	headOut, headIn, kt, kh, kw, err := hw.conv3d("head")
	if err != nil {
		return plan, err
	}
	if headIn != channels || kt != 3 || kh != 3 || kw != 3 {
		return plan, fmt.Errorf("vae decoder: decoder.head.2.weight shape %v incompatible with channels=%d", hw.shape, channels)
	}
	if err := compiler.vector("decoder.head.0.gamma", channels); err != nil {
		return plan, err
	}
	if err := compiler.vector("decoder.head.2.bias", headOut); err != nil {
		return plan, err
	}
	compiler.add(vaeOpHead, "decoder.head", channels, headOut, "decoder.head.0.gamma", "decoder.head.2.weight", "decoder.head.2.bias")
	plan.OutputChannels = headOut
	plan.ops = compiler.ops

	// Chain continuity across the compiled graph.
	for index := 1; index < len(plan.ops); index++ {
		if plan.ops[index].cIn != plan.ops[index-1].cOut {
			return plan, fmt.Errorf("vae decoder: op %d (%s) input channels=%d, previous output=%d",
				index, plan.ops[index].prefix, plan.ops[index].cIn, plan.ops[index-1].cOut)
		}
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

func vaeOpTensorCount(op vaeDecoderOp) int {
	switch op.kind {
	case vaeOpPointwise, vaeOpConv, vaeOpDownsample2D, vaeOpUpsample2D:
		return 2
	case vaeOpResidual:
		if op.cIn != op.cOut {
			return 8
		}
		return 6
	case vaeOpAttention:
		return 5
	case vaeOpDownsample3D, vaeOpUpsample3D:
		return 4
	case vaeOpHead:
		return 3
	}
	return 0
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
	nextFrames := min(2, frames)
	if replicatePrefix || (frames < 2 && prior.initialized && prior.frames > 0) {
		nextFrames = frames + 1
	}
	out := make([]float32, c*nextFrames*spatial)
	for ch := 0; ch < c; ch++ {
		for ti := 0; ti < nextFrames; ti++ {
			dst := out[(ch*nextFrames+ti)*spatial : (ch*nextFrames+ti+1)*spatial]
			switch {
			case replicatePrefix && ti == 0:
				// zero prefix (already zeroed)
			case replicatePrefix:
				copy(dst, x[(ch*frames+ti-1)*spatial:])
			case frames >= 2:
				copy(dst, x[(ch*frames+frames-nextFrames+ti)*spatial:])
			case prior.frames > 0 && ti == 0:
				copy(dst, prior.data[(ch*prior.frames+prior.frames-1)*spatial:])
			default:
				src := ti
				if prior.frames > 0 {
					src = ti - 1
				}
				copy(dst, x[(ch*frames+src)*spatial:])
			}
		}
	}
	return vaeTemporalCache{data: out, frames: nextFrames, initialized: true}
}

// channelRMSNormInto: RMS over channels at each position, sqrt(C)-scaled,
// zero-guarded (reference F.normalize(dim=1)*sqrt(C)*gamma).
func channelRMSNormInto(out, x, gamma []float32, c, plane int) error {
	if c <= 0 || plane <= 0 {
		return fmt.Errorf("vae rms norm: bad shape c=%d plane=%d", c, plane)
	}
	if len(x) != c*plane || len(out) != len(x) || len(gamma) != c {
		return fmt.Errorf("vae rms norm: bad lengths out=%d x=%d gamma=%d", len(out), len(x), len(gamma))
	}
	scale := math.Sqrt(float64(c))
	for pos := 0; pos < plane; pos++ {
		var sumSq float64
		for ch := 0; ch < c; ch++ {
			v := float64(x[ch*plane+pos])
			sumSq += v * v
		}
		norm := math.Sqrt(sumSq)
		if norm < channelNormZeroGuard {
			norm = channelNormZeroGuard
		}
		for ch := 0; ch < c; ch++ {
			idx := ch*plane + pos
			out[idx] = float32(float64(x[idx]) / norm * scale * float64(gamma[ch]))
		}
	}
	return nil
}

// spatialAttentionInto: per-frame spatial self-attention over qkv
// [3c][t][h*w] planes, residual added by the caller.
func spatialAttentionInto(out, qkv []float32, c, t, frame int) error {
	if c <= 0 || t <= 0 || frame <= 0 || len(qkv) != 3*c*t*frame || len(out) != c*t*frame {
		return fmt.Errorf("vae attention: bad lengths out=%d qkv=%d", len(out), len(qkv))
	}
	scale := 1 / math.Sqrt(float64(c))
	scores := make([]float32, frame)
	kOff := c * t * frame
	vOff := 2 * c * t * frame
	for ti := 0; ti < t; ti++ {
		for qi := 0; qi < frame; qi++ {
			for kj := 0; kj < frame; kj++ {
				var dot float64
				for ch := 0; ch < c; ch++ {
					qIdx := (ch*t+ti)*frame + qi
					kIdx := kOff + (ch*t+ti)*frame + kj
					dot += float64(qkv[qIdx]) * float64(qkv[kIdx])
				}
				scores[kj] = float32(dot * scale)
			}
			hostmath.SoftmaxInPlace(scores)
			for ch := 0; ch < c; ch++ {
				var acc float64
				for kj, p := range scores {
					vIdx := vOff + (ch*t+ti)*frame + kj
					acc += float64(p) * float64(qkv[vIdx])
				}
				out[(ch*t+ti)*frame+qi] = float32(acc)
			}
		}
	}
	return nil
}

// timeInterleaveInto: temporal upsample reshape — the time conv's doubled
// channel output [2c][t][s] interleaves to [c][2t][s], even output frames
// from the first channel half (reference reshape+stack).
func timeInterleaveInto(out, convolved []float32, c, frames, spatial int) error {
	if len(convolved) != 2*c*frames*spatial || len(out) != len(convolved) {
		return fmt.Errorf("vae time interleave: bad lengths out=%d convolved=%d", len(out), len(convolved))
	}
	outFrames := 2 * frames
	for ch := 0; ch < c; ch++ {
		for ti := 0; ti < outFrames; ti++ {
			src := ((ti&1)*c + ch) * frames * spatial
			copy(out[(ch*outFrames+ti)*spatial:(ch*outFrames+ti+1)*spatial],
				convolved[src+(ti/2)*spatial:])
		}
	}
	return nil
}

// vaeLoadedOp: an op with weights resident (gammas flattened to channels).
type vaeLoadedOp struct {
	vaeDecoderOp
	values [][]float32
}

func loadVAEDecoderOps(reader *pytorchzip.Reader, plan VAEDecoderPlan) ([]vaeLoadedOp, error) {
	ops := make([]vaeLoadedOp, len(plan.ops))
	for index, op := range plan.ops {
		values, err := reader.ReadBindingValues(op.bindings)
		if err != nil {
			return nil, fmt.Errorf("vae decoder %s: %w", op.prefix, err)
		}
		ops[index] = vaeLoadedOp{vaeDecoderOp: op, values: values}
	}
	return ops, nil
}

func causalGeometry(cIn, cOut, kt, kh, kw, padT, t, h, w int) hostmath.Conv3DShape {
	return hostmath.Conv3DShape{
		CIn: cIn, COut: cOut, InT: t, InH: h, InW: w,
		KT: kt, KH: kh, KW: kw,
		PadT: padT, PadH: kh / 2, PadW: kw / 2,
		StrideT: 1, StrideH: 1, StrideW: 1,
	}
}

// runVAEOp executes one op on one chunk [cIn][frames][h][w], mutating the
// op's temporal state exactly as the reference chunk-major graph does.
func runVAEOp(op vaeLoadedOp, state *vaeOpState, chunkIndex int, x []float32, frames, h, w int) ([]float32, int, int, int, error) {
	spatial := h * w
	c := op.cIn
	cachedConv := func(out, input []float32, cache *vaeTemporalCache, weight, bias []float32, cOut, kt, kh, kw int) error {
		shape := causalGeometry(c, cOut, kt, kh, kw, kt/2, frames, h, w)
		if err := hostmath.CausalConv3DInto(out, input, cache.data, weight, bias, cache.frames, shape); err != nil {
			return err
		}
		if shape.PadT != 0 {
			*cache = nextTemporalCache(*cache, input, c, frames, spatial, false)
		}
		return nil
	}
	switch op.kind {
	case vaeOpPointwise, vaeOpConv:
		weight, bias := op.values[0], op.values[1]
		kt := 1
		if op.kind == vaeOpConv {
			kt = 3
		}
		out := make([]float32, op.cOut*frames*spatial)
		if err := cachedConv(out, x, &state.cache0, weight, bias, op.cOut, kt, kt, kt); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case vaeOpResidual:
		gamma0, w0, b0 := op.values[0], op.values[1], op.values[2]
		gamma1, w1, b1 := op.values[3], op.values[4], op.values[5]
		n0 := make([]float32, len(x))
		if err := channelRMSNormInto(n0, x, gamma0, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n0)
		h0 := make([]float32, op.cOut*frames*spatial)
		if err := cachedConv(h0, n0, &state.cache0, w0, b0, op.cOut, 3, 3, 3); err != nil {
			return nil, 0, 0, 0, err
		}
		n1 := make([]float32, len(h0))
		if err := channelRMSNormInto(n1, h0, gamma1, op.cOut, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(n1)
		out := make([]float32, len(h0))
		shape1 := causalGeometry(op.cOut, op.cOut, 3, 3, 3, 1, frames, h, w)
		if err := hostmath.CausalConv3DInto(out, n1, state.cache1.data, w1, b1, state.cache1.frames, shape1); err != nil {
			return nil, 0, 0, 0, err
		}
		state.cache1 = nextTemporalCache(state.cache1, n1, op.cOut, frames, spatial, false)
		if op.cIn == op.cOut {
			for i := range out {
				out[i] += x[i]
			}
			return out, frames, h, w, nil
		}
		shortcut := make([]float32, len(out))
		if err := hostmath.ChannelMixF64Into(shortcut, x, op.values[6], op.values[7], c, op.cOut, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		for i := range out {
			out[i] += shortcut[i]
		}
		return out, frames, h, w, nil
	case vaeOpAttention:
		gamma, qkvW, qkvB := op.values[0], op.values[1], op.values[2]
		projW, projB := op.values[3], op.values[4]
		norm := make([]float32, len(x))
		if err := channelRMSNormInto(norm, x, gamma, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		qkv := make([]float32, 3*c*frames*spatial)
		if err := hostmath.ChannelMixF64Into(qkv, norm, qkvW, qkvB, c, 3*c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		if err := spatialAttentionInto(norm, qkv, c, frames, spatial); err != nil {
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
	case vaeOpUpsample2D:
		weight, bias := op.values[0], op.values[1]
		out := make([]float32, op.cOut*frames*vaeSpatialScale*h*vaeSpatialScale*w)
		if err := hostmath.ResizeConv2DInto(out, x, weight, bias, c, op.cOut, frames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, vaeSpatialScale * h, vaeSpatialScale * w, nil
	case vaeOpUpsample3D:
		timeW, timeB := op.values[0], op.values[1]
		resampleW, resampleB := op.values[2], op.values[3]
		spatialInput, spatialFrames := x, frames
		if chunkIndex == 0 {
			state.cache0 = vaeTemporalCache{initialized: true, rep: true}
		} else {
			replicatePrefix := state.cache0.rep && frames < 2
			nextCache := nextTemporalCache(state.cache0, x, c, frames, spatial, replicatePrefix)
			cacheData, cacheFrames := state.cache0.data, state.cache0.frames
			if state.cache0.rep {
				cacheData, cacheFrames = nil, 0
			}
			timeShape := causalGeometry(c, 2*c, 3, 1, 1, 1, frames, h, w)
			timeOut := make([]float32, 2*c*frames*spatial)
			if err := hostmath.CausalConv3DInto(timeOut, x, cacheData, timeW, timeB, cacheFrames, timeShape); err != nil {
				return nil, 0, 0, 0, err
			}
			spatialFrames = 2 * frames
			spatialInput = make([]float32, c*spatialFrames*spatial)
			if err := timeInterleaveInto(spatialInput, timeOut, c, frames, spatial); err != nil {
				return nil, 0, 0, 0, err
			}
			state.cache0 = nextCache
		}
		out := make([]float32, op.cOut*spatialFrames*vaeSpatialScale*h*vaeSpatialScale*w)
		if err := hostmath.ResizeConv2DInto(out, spatialInput, resampleW, resampleB, c, op.cOut, spatialFrames, h, w); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, spatialFrames, vaeSpatialScale * h, vaeSpatialScale * w, nil
	case vaeOpHead:
		gamma, weight, bias := op.values[0], op.values[1], op.values[2]
		norm := make([]float32, len(x))
		if err := channelRMSNormInto(norm, x, gamma, c, frames*spatial); err != nil {
			return nil, 0, 0, 0, err
		}
		hostmath.SiLUInPlace(norm)
		out := make([]float32, op.cOut*frames*spatial)
		if err := cachedConv(out, norm, &state.cache0, weight, bias, op.cOut, 3, 3, 3); err != nil {
			return nil, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	}
	return nil, 0, 0, 0, fmt.Errorf("vae decoder: unsupported op %s", op.kind)
}

// VideoDecodeShape: output extents for a latent volume under the plan's
// derived strides (frames = (F-1)*strideT + 1; the first latent frame decodes
// to a single frame, every later one to strideT frames).
func VideoDecodeShape(plan VAEDecoderPlan, latentFrames, latentH, latentW int) (channels, frames, height, width int, err error) {
	if latentFrames <= 0 || latentH <= 0 || latentW <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("vae decode shape: bad latent %dx%dx%d", latentFrames, latentH, latentW)
	}
	if plan.ZDim <= 0 || plan.OutputChannels <= 0 || plan.Stride[0] <= 0 || plan.Stride[1] <= 0 || plan.Stride[2] <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("vae decode shape: invalid plan %+v", plan.Stride)
	}
	return plan.OutputChannels,
		(latentFrames-1)*plan.Stride[0] + 1,
		latentH * plan.Stride[1],
		latentW * plan.Stride[2],
		nil
}

// DecodeLatentVideo streams the decode one latent frame at a time: denorm
// (z*std+mean), the compiled op chain with per-op temporal caches, clamp to
// [-1,1], then per-frame emission through the sink. Weights load once and
// stay resident (the reference chunk-major graph residency); peak host
// memory is weights plus one chunk's activations.
func DecodeLatentVideo(checkpoint string, plan VAEDecoderPlan, stats VAELatentStats, z []float32, latentFrames, latentH, latentW int, sink VideoFrameSink) (VAEDecodeStats, error) {
	var decodeStats VAEDecodeStats
	if sink == nil {
		return decodeStats, fmt.Errorf("vae decode: nil frame sink")
	}
	if len(plan.ops) == 0 {
		return decodeStats, fmt.Errorf("vae decode: empty plan")
	}
	if len(stats.Mean) != plan.ZDim || len(stats.Std) != plan.ZDim {
		return decodeStats, fmt.Errorf("vae decode: latent stats mean=%d std=%d want %d", len(stats.Mean), len(stats.Std), plan.ZDim)
	}
	for channel, std := range stats.Std {
		if std <= 0 {
			return decodeStats, fmt.Errorf("vae decode: nonpositive std for channel %d", channel)
		}
	}
	spatial := latentH * latentW
	if latentFrames <= 0 || spatial <= 0 || len(z) != plan.ZDim*latentFrames*spatial {
		return decodeStats, fmt.Errorf("vae decode: latent len=%d want %d", len(z), plan.ZDim*latentFrames*spatial)
	}
	outputChannels, totalFrames, outputH, outputW, err := VideoDecodeShape(plan, latentFrames, latentH, latentW)
	if err != nil {
		return decodeStats, err
	}
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
	decodeStats.Ops = len(ops)
	decodeStats.LatentFrames = latentFrames
	decodeStats.WeightBytesRead = plan.UsedWeightBytes
	decodeStats.MaxOpWeightBytes = plan.LargestOpWeightBytes
	decodeStats.OutputChannels, decodeStats.OutputHeight, decodeStats.OutputWidth = outputChannels, outputH, outputW
	decodeStats.Engine = "host_streamed_chunks"
	states := make([]vaeOpState, len(ops))
	frameScratch := make([]float32, 0)
	frameIndex := 0
	for chunkIndex := 0; chunkIndex < latentFrames; chunkIndex++ {
		// Denormalized chunk [z][1][h][w].
		x := make([]float32, plan.ZDim*spatial)
		for ch := 0; ch < plan.ZDim; ch++ {
			mean, std := stats.Mean[ch], stats.Std[ch]
			src := z[(ch*latentFrames+chunkIndex)*spatial:]
			dst := x[ch*spatial:]
			for pos := 0; pos < spatial; pos++ {
				dst[pos] = src[pos]*std + mean
			}
		}
		frames, h, w := 1, latentH, latentW
		for opIndex := range ops {
			x, frames, h, w, err = runVAEOp(ops[opIndex], &states[opIndex], chunkIndex, x, frames, h, w)
			if err != nil {
				return decodeStats, fmt.Errorf("vae decode %s chunk %d: %w", ops[opIndex].prefix, chunkIndex, err)
			}
			samplePeak()
		}
		if h != outputH || w != outputW {
			return decodeStats, fmt.Errorf("vae decode chunk %d output %dx%d, want %dx%d", chunkIndex, w, h, outputW, outputH)
		}
		for i, v := range x {
			switch {
			case v < -1:
				x[i] = -1
			case v > 1:
				x[i] = 1
			}
		}
		chunkSpatial := h * w
		frameElements := outputChannels * chunkSpatial
		if cap(frameScratch) < frameElements {
			frameScratch = make([]float32, frameElements)
		}
		frame := frameScratch[:frameElements]
		for chunkFrame := 0; chunkFrame < frames; chunkFrame++ {
			for ch := 0; ch < outputChannels; ch++ {
				copy(frame[ch*chunkSpatial:(ch+1)*chunkSpatial], x[(ch*frames+chunkFrame)*chunkSpatial:])
			}
			if err := sink(frameIndex, frame, h, w); err != nil {
				return decodeStats, fmt.Errorf("vae decode frame sink %d: %w", frameIndex, err)
			}
			frameIndex++
		}
		samplePeak()
	}
	if frameIndex != totalFrames {
		return decodeStats, fmt.Errorf("vae decode produced %d frames, want %d", frameIndex, totalFrames)
	}
	decodeStats.OutputFrames = frameIndex
	decodeStats.DecodeWallSec = time.Since(started).Seconds()
	return decodeStats, nil
}
