package speechrecognition

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/safetensors"
)

// tinyTransducer is a shape/lifecycle fixture, not a numerical model oracle.
// The zero joint weights make token 1 win independently of the encoder; the
// declared symbol limit must therefore force every encoder-frame advance.
func tinyTransducer(t *testing.T) (*safetensors.Source, Declaration, TransducerBinding, *Transducer) {
	t.Helper()
	values := map[string][]float32{}
	shapes := map[string][]int{}
	add := func(name string, shape []int, data []float32) string {
		shapes[name] = shape
		values[name] = data
		return name
	}
	identity := AffineBinding{Weight: add("identity", []int{2, 2}, []float32{1, 0, 0, 1})}
	normalize := AffineBinding{Weight: add("norm.weight", []int{2}, []float32{1, 1}), Bias: add("norm.bias", []int{2}, []float32{0, 0})}
	ff := FeedForwardBinding{Norm: normalize, In: identity, Out: identity}
	block := BlockBinding{First: ff, Second: ff, OutNorm: normalize,
		Attention:   AttentionBinding{Norm: normalize, Query: identity, Key: identity, Value: identity, Out: identity, Heads: 1, BlockFrames: 2, ProjectedRelative: &ProjectedRelativeBinding{Projection: identity, ContentBias: add("content.bias", []int{1, 2}, []float32{0, 0}), PositionBias: add("position.bias", []int{1, 2}, []float32{0, 0}), Base: 10000, LeftContext: 2, MaxPositions: 4}},
		Convolution: ConvolutionBinding{Norm: normalize, In: AffineBinding{Weight: add("glu", []int{4, 2}, []float32{1, 0, 0, 1, 0, 0, 0, 0})}, Kernel: add("depthwise", []int{2, 1, 3}, []float32{0, 0, 1, 0, 0, 1}), LayerNorm: normalize, Out: identity, Stride: 1, Causal: true}}
	d := Declaration{Input: identity, Output: identity, Blocks: []BlockBinding{block}, Activation: "silu", FeedForwardScale: .5, LayerNormEpsilon: 1e-5, BatchNormEpsilon: 1e-5}
	b := TransducerBinding{Bands: 2, Subsampling: []SpatialConvolutionBinding{{Affine: AffineBinding{Weight: add("spatial", []int{1, 1, 1, 1}, []float32{1})}, Stride: [2]uint32{1, 1}}},
		ConditionIn: AffineBinding{Weight: add("condition", []int{2, 3}, []float32{1, 0, 0, 0, 1, 0})}, ConditionOut: identity, ConditionSlots: 1,
		Embedding: add("embedding", []int{3, 2}, make([]float32, 6)), Recurrent: []RecurrentBinding{{Input: AffineBinding{Weight: add("lstm.input", []int{8, 2}, make([]float32, 16))}, Recurrent: AffineBinding{Weight: add("lstm.recurrent", []int{8, 2}, make([]float32, 16))}}},
		DecoderProjection: identity, Joint: AffineBinding{Weight: add("joint", []int{3, 2}, make([]float32, 6)), Bias: add("joint.bias", []int{3}, []float32{0, 1, -1})}, Blank: 2, MaxSymbolsPerFrame: 2}
	directory := t.TempDir()
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { source.Close() })
	model, err := LoadTransducer(t.Context(), source, d, b, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	return source, d, b, model
}

func TestTransducerStateIsolationAndForcedAdvance(t *testing.T) {
	_, _, _, model := tinyTransducer(t)
	x := []float32{1, -1, 2, -2}
	wantTokens, wantDurations := []int{2, 1, 1, 1, 1}, []int{0, 0, 1, 0, 1}
	var offline, a, b TransducerWorkspace
	result, err := model.Recognize(t.Context(), x, 2, &offline, nil)
	if err != nil || !slices.Equal(result.Tokens, wantTokens) || !slices.Equal(result.Durations, wantDurations) {
		t.Fatalf("forced progress=%+v: %v", result, err)
	}
	first, err := model.RecognizeChunk(t.Context(), x, 2, false, &a, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(first.Tokens, wantTokens) || !slices.Equal(first.Durations, wantDurations) {
		t.Fatal("first chunk differs")
	}
	for range 8 {
		if _, err := model.RecognizeChunk(t.Context(), x, 2, false, &a, nil); err != nil {
			t.Fatal(err)
		}
	}
	if a.stream.encoder.attention[0].frames != model.encoder.blocks[0].attention.leftContext {
		t.Fatal("cache was not bounded")
	}
	second, err := model.RecognizeChunk(t.Context(), x, 2, true, &b, nil)
	if err != nil || !slices.Equal(second.Tokens, wantTokens) {
		t.Fatalf("isolated stream differs: %v", err)
	}
	if _, err := model.RecognizeChunk(t.Context(), x, 2, false, &b, nil); err == nil {
		t.Fatal("finalized stream accepted")
	}
	if _, err := model.Recognize(t.Context(), x, 2, &a, nil); err == nil {
		t.Fatal("stream reused as offline")
	}
	if _, err := model.RecognizeChunk(t.Context(), x, 2, false, &offline, nil); err == nil {
		t.Fatal("offline reused as stream")
	}
	if !slices.Equal(x, []float32{1, -1, 2, -2}) {
		t.Fatal("mutated caller features")
	}
	a.steps = math.MaxInt
	if _, err := model.RecognizeChunk(t.Context(), x, 2, false, &a, nil); err == nil {
		t.Fatal("overflowing emission position accepted")
	}
	if _, err := model.Recognize(t.Context(), make([]float32, 10), 5, &TransducerWorkspace{}, nil); err == nil {
		t.Fatal("declared position limit ignored")
	}
}

func TestProjectedAttentionDeclarationRefusals(t *testing.T) {
	source, d, b, _ := tinyTransducer(t)
	for _, tc := range []struct {
		name   string
		change func(*BlockBinding)
	}{
		{"position limit", func(b *BlockBinding) { b.Attention.ProjectedRelative.MaxPositions = 0 }},
		{"negative context", func(b *BlockBinding) { b.Attention.ProjectedRelative.LeftContext = -1 }},
		{"nonfinite base", func(b *BlockBinding) { b.Attention.ProjectedRelative.Base = math.NaN() }},
		{"mixed position operations", func(b *BlockBinding) { b.Attention.Relative = "identity" }},
		{"mixed normalization", func(b *BlockBinding) { b.Convolution.BatchNorm = b.Convolution.LayerNorm }},
		{"causal pooling", func(b *BlockBinding) { b.Convolution.Stride = 2 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := d
			candidate.Blocks = slices.Clone(d.Blocks)
			position := *d.Blocks[0].Attention.ProjectedRelative
			candidate.Blocks[0].Attention.ProjectedRelative = &position
			tc.change(&candidate.Blocks[0])
			if _, err := LoadTransducer(t.Context(), source, candidate, b, 1<<20); err == nil {
				t.Fatal("invalid operation mixture admitted")
			}
		})
	}
}

func TestTransducerAdmissionAndCancellation(t *testing.T) {
	source, d, b, model := tinyTransducer(t)
	x := []float32{1, -1, 2, -2}
	for _, change := range []func(*TransducerBinding){
		func(b *TransducerBinding) { b.ConditionIndex = b.ConditionSlots },
		func(b *TransducerBinding) { b.Blank = 3 },
		func(b *TransducerBinding) { b.MaxSymbolsPerFrame = 0 },
		func(b *TransducerBinding) { b.Bands++ },
		func(b *TransducerBinding) { b.Embedding = "absent" },
	} {
		candidate := b
		change(&candidate)
		if _, err := LoadTransducer(t.Context(), source, d, candidate, 1<<20); err == nil {
			t.Fatal("invalid binding admitted")
		}
	}
	if _, err := LoadTransducer(t.Context(), source, d, b, model.weightBytes-1); err == nil {
		t.Fatal("under-budget weights admitted")
	}
	limited, err := LoadTransducer(t.Context(), source, d, b, model.weightBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := limited.Recognize(t.Context(), x, 2, &TransducerWorkspace{}, nil); err == nil {
		t.Fatal("unbudgeted workspace admitted")
	}
	for _, candidate := range []*Transducer{nil, {}} {
		if _, err := candidate.RecognizeChunk(t.Context(), x, 2, false, &TransducerWorkspace{}, nil); err == nil {
			t.Fatal("empty executor admitted")
		}
	}
	for _, features := range [][]float32{x[:2], {1, float32(math.NaN()), 2, 3}} {
		if _, err := model.RecognizeChunk(t.Context(), features, 2, false, &TransducerWorkspace{}, nil); err == nil {
			t.Fatal("invalid features admitted")
		}
	}
	var state TransducerWorkspace
	ctx, cancel := context.WithCancelCause(t.Context())
	_, err = model.RecognizeChunk(ctx, x, 2, false, &state, &TransducerObserver{Decoder: func(DecoderTrace) error { cancel(nil); return ctx.Err() }})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("decoder cancellation=%v", err)
	}
	if _, err := model.RecognizeChunk(t.Context(), x, 2, false, &state, nil); err == nil {
		t.Fatal("partially executed decoder resumed")
	}
	ctx, cancel = context.WithCancelCause(t.Context())
	cancel(nil)
	var untouched TransducerWorkspace
	if _, err := model.RecognizeChunk(ctx, x, 2, false, &untouched, nil); !errors.Is(err, context.Canceled) || untouched.stream != nil {
		t.Fatal("pre-cancelled invocation allocated state")
	}
}
