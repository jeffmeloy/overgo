package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/safetensors"
)

type linear struct {
	weight, bias []float32
	in, out      int
}
type norm struct{ weight, bias []float32 }
type feedForward struct {
	norm    norm
	in, out linear
}
type attention struct {
	norm                        norm
	query, key, value, out      linear
	relative                    []float32
	heads, width, block, radius int
}
type convolution struct {
	norm                                norm
	in, out                             linear
	kernel, scale, bias, mean, variance []float32
	channels, kernelSize, stride        int
}
type block struct {
	first, second feedForward
	attention     attention
	convolution   convolution
	out           norm
}

// Encoder owns immutable promoted weights and validated tensor-derived geometry.
// A Workspace is required per concurrent call. No source handles are retained.
type Encoder struct {
	input, output, feedback  linear
	blocks                   []block
	feedbackAfter            int
	activation               string
	ffScale                  float32
	lnEpsilon, bnEpsilon     float64
	weightBytes, memoryBytes uint64
}

type loader struct {
	source        *safetensors.Source
	values        map[string][]float32
	bytes, budget uint64
}

func (l *loader) tensor(ctx context.Context, name string, dimensions ...int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t, found := l.source.Tensors[name]
	if !found || len(t.Shape) != len(dimensions) {
		return nil, fmt.Errorf("tensor %q absent or wrong rank", name)
	}
	for i, d := range dimensions {
		if d <= 0 || uint64(d) != t.Shape[i] {
			return nil, fmt.Errorf("tensor %q shape %v differs from %v", name, t.Shape, dimensions)
		}
	}
	if v, ok := l.values[name]; ok {
		return v, nil
	}
	if t.Elements() > (l.budget-l.bytes)/binaryschema.Uint32Bytes {
		return nil, errors.New("encoder weights exceed byte budget")
	}
	v, err := safetensors.ReadF32(t)
	if err != nil {
		return nil, err
	}
	for _, x := range v {
		if !checked.Finite32(x) {
			return nil, fmt.Errorf("tensor %q contains non-finite values", name)
		}
	}
	l.bytes += uint64(len(v)) * binaryschema.Uint32Bytes
	l.values[name] = v
	return v, nil
}

func (l *loader) shape(ctx context.Context, name string, rank int) ([]int, error) {
	t, ok := l.source.Tensors[name]
	if !ok || len(t.Shape) != rank {
		return nil, fmt.Errorf("tensor %q absent or wrong rank", name)
	}
	d := make([]int, rank)
	for i, n := range t.Shape {
		v, ok := checked.Int(n)
		if !ok || v <= 0 {
			return nil, fmt.Errorf("tensor %q invalid extent", name)
		}
		d[i] = v
	}
	return d, nil
}

func (l *loader) linear(ctx context.Context, b AffineBinding, input int) (linear, error) {
	d, err := l.shape(ctx, b.Weight, 2)
	if err != nil {
		return linear{}, err
	}
	if input > 0 && d[1] != input {
		return linear{}, fmt.Errorf("projection %q input %d, want %d", b.Weight, d[1], input)
	}
	x := linear{in: d[1], out: d[0]}
	x.weight, err = l.tensor(ctx, b.Weight, d...)
	if err != nil {
		return linear{}, err
	}
	if b.Bias != "" {
		x.bias, err = l.tensor(ctx, b.Bias, x.out)
	}
	return x, err
}

func (l *loader) norm(ctx context.Context, b AffineBinding, width int) (norm, error) {
	w, err := l.tensor(ctx, b.Weight, width)
	if err != nil {
		return norm{}, err
	}
	bias, err := l.tensor(ctx, b.Bias, width)
	return norm{w, bias}, err
}

func (l *loader) feedForward(ctx context.Context, b FeedForwardBinding, h int) (feedForward, error) {
	var f feedForward
	var err error
	if f.norm, err = l.norm(ctx, b.Norm, h); err != nil {
		return f, err
	}
	if f.in, err = l.linear(ctx, b.In, h); err != nil {
		return f, err
	}
	if f.out, err = l.linear(ctx, b.Out, f.in.out); err != nil {
		return f, err
	}
	if f.out.out != h {
		return f, errors.New("feed-forward residual width differs")
	}
	return f, nil
}

// LoadEncoder resolves declared roles through the existing Safetensors owner.
// The byte budget covers promoted weight and workspace numeric backing arrays,
// not caller-owned inputs, source metadata, headers or allocator overhead.
func LoadEncoder(ctx context.Context, source *safetensors.Source, d Declaration, memoryBytes uint64) (*Encoder, error) {
	if ctx == nil || source == nil || len(d.Blocks) == 0 || memoryBytes == 0 ||
		!checked.PositiveFinite64(d.LayerNormEpsilon) || !checked.PositiveFinite64(d.BatchNormEpsilon) ||
		!checked.Finite32(d.FeedForwardScale) || d.FeedbackAfter < 0 || d.FeedbackAfter > len(d.Blocks) ||
		(d.Activation != "silu" && d.Activation != "gelu-erf") {
		return nil, errors.New("encoder: invalid execution or declaration")
	}
	if d.FeedbackAfter == 0 && (d.Feedback.Weight != "" || d.Feedback.Bias != "") {
		return nil, errors.New("encoder: feedback weights without a feedback boundary")
	}
	l := loader{source: source, values: make(map[string][]float32), budget: memoryBytes}
	e := &Encoder{memoryBytes: memoryBytes, feedbackAfter: d.FeedbackAfter, activation: d.Activation, ffScale: d.FeedForwardScale, lnEpsilon: d.LayerNormEpsilon, bnEpsilon: d.BatchNormEpsilon}
	var err error
	if e.input, err = l.linear(ctx, d.Input, 0); err != nil {
		return nil, err
	}
	h := e.input.out
	if e.output, err = l.linear(ctx, d.Output, h); err != nil {
		return nil, err
	}
	if d.FeedbackAfter > 0 {
		if e.feedback, err = l.linear(ctx, d.Feedback, e.output.out); err != nil {
			return nil, err
		}
		if e.feedback.out != h {
			return nil, errors.New("encoder: posterior feedback width differs")
		}
	}
	for i, b := range d.Blocks {
		layer, err := l.block(ctx, b, h)
		if err != nil {
			return nil, fmt.Errorf("encoder block %d: %w", i, err)
		}
		e.blocks = append(e.blocks, layer)
	}
	e.weightBytes = l.bytes
	return e, nil
}

func (l *loader) block(ctx context.Context, b BlockBinding, h int) (block, error) {
	var x block
	var err error
	if x.first, err = l.feedForward(ctx, b.First, h); err != nil {
		return x, err
	}
	if x.second, err = l.feedForward(ctx, b.Second, h); err != nil {
		return x, err
	}
	if x.out, err = l.norm(ctx, b.OutNorm, h); err != nil {
		return x, err
	}
	a := &x.attention
	if b.Attention.Heads <= 0 || b.Attention.BlockFrames <= 0 {
		return x, errors.New("invalid attention partition")
	}
	if a.norm, err = l.norm(ctx, b.Attention.Norm, h); err != nil {
		return x, err
	}
	if a.query, err = l.linear(ctx, b.Attention.Query, h); err != nil {
		return x, err
	}
	if a.key, err = l.linear(ctx, b.Attention.Key, h); err != nil {
		return x, err
	}
	if a.value, err = l.linear(ctx, b.Attention.Value, h); err != nil {
		return x, err
	}
	if a.out, err = l.linear(ctx, b.Attention.Out, a.query.out); err != nil {
		return x, err
	}
	if a.query.out%b.Attention.Heads != 0 || a.key.out != a.query.out || a.value.out != a.query.out || a.out.out != h || len(a.query.bias) != 0 || len(a.key.bias) != 0 || len(a.value.bias) != 0 {
		return x, errors.New("attention requires equal-width bias-free query/key/value projections")
	}
	a.heads, a.width, a.block = b.Attention.Heads, a.query.out/b.Attention.Heads, b.Attention.BlockFrames
	d, err := l.shape(ctx, b.Attention.Relative, 2)
	if err != nil {
		return x, err
	}
	if d[0]%2 != 1 || d[1] != a.width || a.block-1 > d[0]/2 {
		return x, errors.New("relative position table does not cover block distances")
	}
	a.radius = d[0] / 2
	if a.relative, err = l.tensor(ctx, b.Attention.Relative, d...); err != nil {
		return x, err
	}
	c := &x.convolution
	if b.Convolution.Stride <= 0 {
		return x, errors.New("invalid convolution stride")
	}
	c.stride = b.Convolution.Stride
	if c.norm, err = l.norm(ctx, b.Convolution.Norm, h); err != nil {
		return x, err
	}
	if c.in, err = l.linear(ctx, b.Convolution.In, h); err != nil {
		return x, err
	}
	if c.in.out%2 != 0 {
		return x, errors.New("GLU projection must contain paired channels")
	}
	c.channels = c.in.out / 2
	if c.out, err = l.linear(ctx, b.Convolution.Out, c.channels); err != nil {
		return x, err
	}
	if c.out.out != h {
		return x, errors.New("convolution residual width differs")
	}
	d, err = l.shape(ctx, b.Convolution.Kernel, 3)
	if err != nil {
		return x, err
	}
	if !slices.Equal(d[:2], []int{c.channels, 1}) || d[2]%2 != 1 {
		return x, errors.New("centered depthwise kernel must have odd width")
	}
	c.kernelSize = d[2]
	if c.kernel, err = l.tensor(ctx, b.Convolution.Kernel, d...); err != nil {
		return x, err
	}
	bn, err := l.norm(ctx, b.Convolution.BatchNorm, c.channels)
	if err != nil {
		return x, err
	}
	c.scale, c.bias = bn.weight, bn.bias
	if c.mean, err = l.tensor(ctx, b.Convolution.Mean, c.channels); err != nil {
		return x, err
	}
	if c.variance, err = l.tensor(ctx, b.Convolution.Variance, c.channels); err != nil {
		return x, err
	}
	for _, v := range c.variance {
		if v < 0 || math.IsInf(float64(v), 0) {
			return x, errors.New("negative batch normalization variance")
		}
	}
	return x, nil
}
