package speechrecognition

// Acoustic speaker-activity composition follows audio.cpp
// 3497b7cc44753e2c141d8fe60ac42cec433e3281, Copyright 2026 ShugoAI LLC,
// Apache-2.0 (licenses/audio.cpp.txt). Changes: declaration-bound tensor roles,
// shared Go arithmetic, valid-frame execution without padded keys, reusable
// bounded workspaces, context cancellation and explicit failure contracts.

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
	"overgo/internal/scratch"
)

// PostNormBlockBinding declares full bidirectional attention and a ReLU
// feed-forward residual, each followed by affine layer normalization.
// Query and key are each scaled by the inverse fourth root of head width.
type PostNormBlockBinding struct {
	Query         AffineBinding      `json:"query"`
	Key           AffineBinding      `json:"key"`
	Value         AffineBinding      `json:"value"`
	Out           AffineBinding      `json:"out"`
	AttentionNorm AffineBinding      `json:"attention_norm"`
	FeedForward   FeedForwardBinding `json:"feed_forward"`
}

// ActivityBinding declares acoustic subsampling, post-normalized attention and
// a ReLU/affine/ReLU/affine/sigmoid speaker-activity head. Tensor names and
// geometry are explicit inputs; no model family is inferred.
type ActivityBinding struct {
	Bands            int                         `json:"bands"`
	Subsampling      []SpatialConvolutionBinding `json:"subsampling"`
	Blocks           []PostNormBlockBinding      `json:"blocks"`
	Heads            int                         `json:"heads"`
	MaxFrames        int                         `json:"max_frames"`
	LayerNormEpsilon float64                     `json:"layer_norm_epsilon"`
	Hidden           AffineBinding               `json:"hidden"`
	Output           AffineBinding               `json:"output"`
}

type postNormBlock struct {
	query, key, value, out linear
	norm                   norm
	feedForward            feedForward
}

// SpeakerActivity owns an immutable acoustic encoder and frame-activity head.
// It predicts probabilities, not speaker identities or calibrated confidence.
type SpeakerActivity struct {
	encoder                  *Encoder
	subsampling              []spatialConvolution
	blocks                   []postNormBlock
	hidden, output           linear
	bands, heads, maxFrames  int
	epsilon                  float64
	weightBytes, memoryBytes uint64
}

// ActivityWorkspace owns one nonconcurrent caller's numeric state. Its zero
// value is usable; results borrow it until the next call.
type ActivityWorkspace struct {
	owner                                                           *SpeakerActivity
	encoder                                                         Workspace
	x, branch, expanded, query, key, value, attended, probabilities []float32
}

// LoadSpeakerActivity validates all declared tensors and composes the existing
// acoustic, spatial-convolution and host attention operations. memoryBytes
// bounds numeric weights and workspace, not allocator overhead or process RSS.
func LoadSpeakerActivity(ctx context.Context, source *safetensors.Source, encoder Declaration, binding ActivityBinding, memoryBytes uint64) (*SpeakerActivity, error) {
	if ctx == nil || source == nil || memoryBytes == 0 || binding.Bands <= 0 || binding.Heads <= 0 ||
		len(binding.Blocks) == 0 || len(binding.Subsampling) == 0 || binding.MaxFrames <= 0 || !checked.PositiveFinite64(binding.LayerNormEpsilon) || encoder.FeedbackAfter != 0 {
		return nil, errors.New("speaker activity: invalid declaration")
	}
	e, err := LoadEncoder(ctx, source, encoder, memoryBytes)
	if err != nil {
		return nil, err
	}
	l := loader{source: source, values: make(map[string][]float32), budget: memoryBytes - e.weightBytes}
	p := &SpeakerActivity{encoder: e, bands: binding.Bands, heads: binding.Heads, maxFrames: binding.MaxFrames, epsilon: binding.LayerNormEpsilon, memoryBytes: memoryBytes}
	p.subsampling, err = l.subsampling(ctx, binding.Subsampling, binding.Bands, e.input.in)
	if err != nil {
		return nil, err
	}
	width := e.output.out
	if width%p.heads != 0 {
		return nil, errors.New("speaker activity: head partition differs")
	}
	for index, b := range binding.Blocks {
		var block postNormBlock
		for _, pair := range []struct {
			binding AffineBinding
			target  *linear
		}{{b.Query, &block.query}, {b.Key, &block.key}, {b.Value, &block.value}, {b.Out, &block.out}} {
			*pair.target, err = l.linear(ctx, pair.binding, width)
			if err != nil {
				return nil, fmt.Errorf("speaker activity block %d: %w", index, err)
			}
			if pair.target.out != width {
				return nil, errors.New("speaker activity: residual projection width differs")
			}
		}
		block.norm, err = l.norm(ctx, b.AttentionNorm, width)
		if err != nil {
			return nil, err
		}
		block.feedForward, err = l.feedForward(ctx, b.FeedForward, width)
		if err != nil {
			return nil, err
		}
		p.blocks = append(p.blocks, block)
	}
	p.hidden, err = l.linear(ctx, binding.Hidden, width)
	if err != nil {
		return nil, err
	}
	p.output, err = l.linear(ctx, binding.Output, p.hidden.out)
	if err != nil {
		return nil, err
	}
	p.weightBytes = e.weightBytes + l.bytes
	e.memoryBytes -= l.bytes
	return p, nil
}

func (w *ActivityWorkspace) buffers() []*[]float32 {
	return []*[]float32{&w.x, &w.branch, &w.expanded, &w.query, &w.key, &w.value, &w.attended, &w.probabilities}
}

func (p *SpeakerActivity) reserve(w *ActivityWorkspace, rows int) (uint64, error) {
	width, expanded := p.encoder.output.out, p.hidden.out
	for _, b := range p.blocks {
		expanded = max(expanded, b.feedForward.in.out)
	}
	widths := [...]int{width, width, expanded, width, width, width, width, p.output.out}
	var counts [len(widths)]int
	var bytes uint64
	for i, buffer := range w.buffers() {
		n, ok := checked.MulInt(rows, widths[i])
		if !ok {
			return 0, errors.New("speaker activity: workspace extent overflows")
		}
		counts[i] = n
		size, ok := checked.Mul64(uint64(max(cap(*buffer), n)), binaryschema.Uint32Bytes)
		if !ok {
			return 0, errors.New("speaker activity: workspace bytes overflow")
		}
		bytes, ok = checked.Add64(bytes, size)
		if !ok || bytes > p.memoryBytes-p.weightBytes {
			return 0, errors.New("speaker activity: workspace exceeds byte budget")
		}
	}
	// The shared attention owner allocates at most one float64 score row per
	// head; reserve all heads even when its worker policy uses fewer at once.
	scoreValues, ok := checked.Mul64(uint64(rows), uint64(p.heads))
	if !ok {
		return 0, errors.New("speaker activity: attention extent overflows")
	}
	scoreBytes, ok := checked.Mul64(scoreValues, binaryschema.Uint64Bytes)
	if !ok {
		return 0, errors.New("speaker activity: attention bytes overflow")
	}
	bytes, ok = checked.Add64(bytes, scoreBytes)
	if !ok || bytes > p.memoryBytes-p.weightBytes {
		return 0, errors.New("speaker activity: attention exceeds byte budget")
	}
	// Include retained encoder arrays before allocating a larger side workspace.
	retained := bytes
	for _, buffer := range w.encoder.buffers() {
		size, valid := checked.Mul64(uint64(cap(*buffer)), binaryschema.Uint32Bytes)
		if !valid {
			return 0, errors.New("speaker activity: retained encoder extent overflows")
		}
		retained, valid = checked.Add64(retained, size)
		if !valid || retained > p.memoryBytes-p.weightBytes {
			return 0, errors.New("speaker activity: retained state exceeds byte budget")
		}
	}
	for i, buffer := range w.buffers() {
		*buffer = scratch.Resize(*buffer, counts[i])
	}
	return retained, nil
}

// Predict processes valid unpadded frontend frames and returns row-major
// speaker probabilities and their frame count. Absent padded keys and GLU
// tails are excluded by construction. It does not provide streaming state or
// preserve speaker labels across independent calls. observe receives encoder
// blocks first, then the output projection and post-normalized blocks.
func (p *SpeakerActivity) Predict(ctx context.Context, features []float32, frames int, w *ActivityWorkspace, observe func(Trace) error) ([]float32, int, error) {
	if p == nil || p.encoder == nil || ctx == nil || w == nil || w.owner != nil && w.owner != p || frames <= 0 {
		return nil, 0, errors.New("speaker activity: invalid execution")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	n, ok := checked.MulInt(frames, p.bands)
	if !ok || n != len(features) {
		return nil, 0, errors.New("speaker activity: feature shape differs")
	}
	for _, value := range features {
		if !checked.Finite32(value) {
			return nil, 0, errors.New("speaker activity: non-finite input")
		}
	}
	for _, buffer := range append(w.buffers(), w.encoder.buffers()...) {
		if checked.SlicesOverlap(features, (*buffer)[:cap(*buffer)]) {
			return nil, 0, errors.New("speaker activity: input aliases workspace")
		}
	}
	// Derive the exact output count before reserving state.
	rows := frames
	for _, stage := range p.subsampling {
		padded, valid := checked.AddInt(rows, int(stage.binding.Padding[2]), int(stage.binding.Padding[3]))
		kernel := int(stage.weight.Shape.Dims[1])
		if !valid || padded < kernel {
			return nil, 0, errors.New("speaker activity: temporal extent invalid")
		}
		rows = (padded-kernel)/int(stage.binding.Stride[1]) + 1
	}
	if rows > p.maxFrames {
		return nil, 0, errors.New("speaker activity: declared frame limit exceeded")
	}
	retained, err := p.reserve(w, rows)
	if err != nil {
		return nil, 0, err
	}
	w.owner = p
	input, actual, _, err := subsample(ctx, p.subsampling, features, frames, p.bands, p.memoryBytes-p.weightBytes-retained, nil)
	if err != nil {
		return nil, 0, err
	}
	if actual != rows {
		return nil, 0, errors.New("speaker activity: planned frame count differs")
	}
	// The encoder separately accounts its own retained buffers; composing
	// numeric state includes the subsampled input and this head's buffers.
	var composed uint64
	for _, buffer := range w.buffers() {
		size, valid := checked.Mul64(uint64(cap(*buffer)), binaryschema.Uint32Bytes)
		if !valid {
			return nil, 0, errors.New("speaker activity: composed state overflows")
		}
		composed, valid = checked.Add64(composed, size)
		if !valid {
			return nil, 0, errors.New("speaker activity: composed state overflows")
		}
	}
	inputBytes, valid := checked.Mul64(uint64(len(input)), binaryschema.Uint32Bytes)
	if !valid {
		return nil, 0, errors.New("speaker activity: composed input overflows")
	}
	scoreValues, valid := checked.Mul64(uint64(rows), uint64(p.heads))
	if !valid {
		return nil, 0, errors.New("speaker activity: composed attention overflows")
	}
	scoreBytes, valid := checked.Mul64(scoreValues, binaryschema.Uint64Bytes)
	if !valid {
		return nil, 0, errors.New("speaker activity: composed attention overflows")
	}
	composed, valid = checked.Add64(composed, inputBytes, scoreBytes)
	if !valid {
		return nil, 0, errors.New("speaker activity: composed state overflows")
	}
	w.encoder.reservedBytes = composed
	hidden, rows, err := p.encoder.Encode(ctx, input, rows, &w.encoder, observe)
	if err != nil {
		return nil, 0, err
	}
	x, err := p.encoder.Project(ctx, hidden, rows, &w.encoder)
	if err != nil {
		return nil, 0, err
	}
	copy(w.x, x)
	x = w.x[:len(x)]
	if err := emitTrace(ctx, observe, len(p.encoder.blocks), rows, p.encoder.output.out, x); err != nil {
		return nil, 0, err
	}
	width := p.encoder.output.out
	headWidth := width / p.heads
	scale := float32(1 / math.Sqrt(math.Sqrt(float64(headWidth))))
	for index, b := range p.blocks {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		b.query.forward(w.query, x, rows)
		b.key.forward(w.key, x, rows)
		b.value.forward(w.value, x, rows)
		for i := range x {
			w.query[i] *= scale
			w.key[i] *= scale
		}
		hostmath.MaskedBidirectionalAttention(w.attended[:len(x)], w.query, w.key, w.value, rows, rows, p.heads, p.heads, headWidth, nil)
		b.out.forward(w.branch, w.attended, rows)
		for i := range x {
			x[i] += w.branch[i]
		}
		hostmath.LayerNormInto(x, x, b.norm.weight, b.norm.bias, rows, width, p.epsilon)
		b.feedForward.in.forward(w.expanded, x, rows)
		for i := range rows * b.feedForward.in.out {
			w.expanded[i] = max(0, w.expanded[i])
		}
		b.feedForward.out.forward(w.branch, w.expanded, rows)
		for i := range x {
			x[i] += w.branch[i]
		}
		hostmath.LayerNormInto(x, x, b.feedForward.norm.weight, b.feedForward.norm.bias, rows, width, p.epsilon)
		if err := emitTrace(ctx, observe, len(p.encoder.blocks)+index+1, rows, width, x); err != nil {
			return nil, 0, err
		}
	}
	for i := range x {
		x[i] = max(0, x[i])
	}
	p.hidden.forward(w.expanded, x, rows)
	for i := range rows * p.hidden.out {
		w.expanded[i] = max(0, w.expanded[i])
	}
	output := w.probabilities[:rows*p.output.out]
	p.output.forward(output, w.expanded, rows)
	for i, value := range output {
		if !checked.Finite32(value) {
			return nil, 0, errors.New("speaker activity: non-finite output")
		}
		output[i] = float32(1 / (1 + math.Exp(-float64(value))))
	}
	return output, rows, ctx.Err()
}
