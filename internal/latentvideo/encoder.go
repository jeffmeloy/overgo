// Relative-position text encoder (T5-family geometry, derived from the
// checkpoint): token embedding rows -> N pre-norm blocks (RMSNorm, relative
// -position self-attention, gated-GELU FFN) -> final RMSNorm. Weights stream
// from the PyTorch ZIP checkpoint one block at a time so peak host memory is
// bounded by a single block, never the 11+GB checkpoint. Ported behavior
// (adaptive latent_video host path), bf16-rounded at the same points.
package latentvideo

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"time"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// EncoderConfig: checkpoint geometry plus explicit profile policy
// (RelativeMaxDistance and NormEps are model facts the checkpoint does not
// carry; everything else derives from tensor shapes).
type EncoderConfig struct {
	VocabSize, Layers, Dim, Heads, FFNDim, RelativeBuckets int
	RelativeMaxDistance                                    int
	NormEps                                                float64
}

// TensorIssue: one contract violation found while compiling a plan.
type TensorIssue struct {
	Name, Want, Got string
}

// EncoderPlan: validated checkpoint contract with runtime bindings.
type EncoderPlan struct {
	OK               bool
	Missing          []TensorIssue
	Unexpected       []TensorIssue
	DTypeMismatches  []TensorIssue
	ShapeMismatches  []TensorIssue
	Config           EncoderConfig
	bindings         []pytorchzip.TensorBinding
	blockWeightBytes []int64
}

const (
	blockNorm1 = iota
	blockQ
	blockK
	blockV
	blockO
	blockNorm2
	blockGate
	blockFC1
	blockFC2
	blockPosition
	blockTensorCount
)

// deriveEncoderConfig reconstructs geometry from tensor shapes; policy
// supplies only RelativeMaxDistance and NormEps.
func deriveEncoderConfig(byName map[string]pytorchzip.TensorMeta, policy EncoderConfig) (EncoderConfig, error) {
	c := EncoderConfig{RelativeMaxDistance: policy.RelativeMaxDistance, NormEps: policy.NormEps}
	norm, ok := byName["norm.weight"]
	if !ok {
		return c, fmt.Errorf("encoder config: norm.weight missing or not 1-d")
	}
	normShape, err := pytorchzip.HostShape(norm, tensor.SingletonExtent)
	if err != nil {
		return c, fmt.Errorf("encoder config: norm.weight: %w", err)
	}
	c.Dim = normShape[tensor.FirstOffset]
	embed, ok := byName["token_embedding.weight"]
	if !ok {
		return c, fmt.Errorf("encoder config: token_embedding.weight missing or incompatible with width=%d", c.Dim)
	}
	embedShape, err := pytorchzip.HostShape(embed, tensor.PairedExtent)
	if err != nil {
		return c, fmt.Errorf("encoder config: token_embedding.weight: %w", err)
	}
	if !checked.Equal(embedShape[tensor.SingletonExtent], c.Dim) {
		return c, fmt.Errorf("encoder config: token_embedding.weight incompatible with width=%d", c.Dim)
	}
	c.VocabSize = embedShape[tensor.FirstOffset]
	pos, ok := byName["blocks.0.pos_embedding.embedding.weight"]
	if !ok {
		return c, fmt.Errorf("encoder config: blocks.0 position embedding missing or not 2-d")
	}
	positionShape, err := pytorchzip.HostShape(pos, tensor.PairedExtent)
	if err != nil {
		return c, fmt.Errorf("encoder config: position embedding: %w", err)
	}
	c.RelativeBuckets, c.Heads = positionShape[tensor.FirstOffset], positionShape[tensor.SingletonExtent]
	fc1, ok := byName["blocks.0.ffn.fc1.weight"]
	if !ok {
		return c, fmt.Errorf("encoder config: blocks.0 ffn.fc1 missing or incompatible with width=%d", c.Dim)
	}
	ffnShape, err := pytorchzip.HostShape(fc1, tensor.PairedExtent)
	if err != nil {
		return c, fmt.Errorf("encoder config: blocks.0 ffn.fc1: %w", err)
	}
	if !checked.Equal(ffnShape[tensor.SingletonExtent], c.Dim) {
		return c, fmt.Errorf("encoder config: blocks.0 ffn.fc1 incompatible with width=%d", c.Dim)
	}
	c.FFNDim = ffnShape[tensor.FirstOffset]
	for layer := tensor.FirstOffset; ; layer++ {
		if _, ok := byName[fmt.Sprintf("blocks.%d.norm1.weight", layer)]; !ok {
			c.Layers = layer
			break
		}
	}
	if !checked.PositiveInts(c.Layers) {
		return c, fmt.Errorf("encoder config: no blocks.N.norm1.weight tensors")
	}
	if _, ok := checked.DivExactInt(c.Dim, c.Heads); !ok {
		return c, fmt.Errorf("encoder config: invalid derived contract %+v", c)
	}
	if !checked.PositiveInts(c.RelativeMaxDistance) || !checked.PositiveFinite64(c.NormEps) {
		return c, fmt.Errorf("encoder config: invalid derived contract %+v", c)
	}
	return c, nil
}

// CompileEncoderPlan validates the full tensor contract and binds it.
func CompileEncoderPlan(metas []pytorchzip.TensorMeta, policy EncoderConfig) (EncoderPlan, error) {
	var plan EncoderPlan
	byName := make(map[string]pytorchzip.TensorMeta, len(metas))
	for _, m := range metas {
		byName[m.Name] = m
	}
	config, configErr := deriveEncoderConfig(byName, policy)
	if configErr != nil {
		plan.Missing = append(plan.Missing, TensorIssue{Name: "config", Want: configErr.Error()})
	} else {
		plan.Config = config
		expected := expectedEncoderSpecs(config)
		for _, m := range metas {
			if _, ok := expected[m.Name]; !ok {
				plan.Unexpected = append(plan.Unexpected, TensorIssue{Name: m.Name, Got: fmt.Sprint(m.Shape)})
			}
		}
		for name, spec := range expected {
			got, ok := byName[name]
			if !ok {
				plan.Missing = append(plan.Missing, TensorIssue{Name: name, Want: fmt.Sprint(spec.shape)})
				continue
			}
			if got.DType != pytorchzip.BFloat16StorageClass() {
				plan.DTypeMismatches = append(plan.DTypeMismatches, TensorIssue{Name: name, Want: pytorchzip.BFloat16StorageClass(), Got: got.DType})
			}
			if !slices.Equal(got.Shape, spec.shape) {
				plan.ShapeMismatches = append(plan.ShapeMismatches, TensorIssue{Name: name, Want: fmt.Sprint(spec.shape), Got: fmt.Sprint(got.Shape)})
			}
		}
	}
	plan.OK = checked.Empty(plan.Missing, plan.Unexpected, plan.DTypeMismatches, plan.ShapeMismatches)
	if plan.OK {
		plan.bindings, plan.blockWeightBytes, configErr = compileEncoderBindings(metas, plan.Config)
		if configErr != nil {
			return plan, configErr
		}
	}
	return plan, nil
}

type tensorSpec struct{ shape []int64 }

func expectedEncoderSpecs(c EncoderConfig) map[string]tensorSpec {
	vocab, dim, dimFFN := int64(c.VocabSize), int64(c.Dim), int64(c.FFNDim)
	numBuckets, numHeads := int64(c.RelativeBuckets), int64(c.Heads)
	specs := map[string]tensorSpec{
		"token_embedding.weight": {shape: []int64{vocab, dim}},
		"norm.weight":            {shape: []int64{dim}},
	}
	for i := range c.Layers {
		p := fmt.Sprintf("blocks.%d.", i)
		specs[p+"norm1.weight"] = tensorSpec{shape: []int64{dim}}
		specs[p+"attn.q.weight"] = tensorSpec{shape: []int64{dim, dim}}
		specs[p+"attn.k.weight"] = tensorSpec{shape: []int64{dim, dim}}
		specs[p+"attn.v.weight"] = tensorSpec{shape: []int64{dim, dim}}
		specs[p+"attn.o.weight"] = tensorSpec{shape: []int64{dim, dim}}
		specs[p+"norm2.weight"] = tensorSpec{shape: []int64{dim}}
		specs[p+"ffn.gate.0.weight"] = tensorSpec{shape: []int64{dimFFN, dim}}
		specs[p+"ffn.fc1.weight"] = tensorSpec{shape: []int64{dimFFN, dim}}
		specs[p+"ffn.fc2.weight"] = tensorSpec{shape: []int64{dim, dimFFN}}
		specs[p+"pos_embedding.embedding.weight"] = tensorSpec{shape: []int64{numBuckets, numHeads}}
	}
	return specs
}

func encoderBlockTensorNames(layer int) []string {
	p := fmt.Sprintf("blocks.%d.", layer)
	return []string{
		p + "norm1.weight",
		p + "attn.q.weight",
		p + "attn.k.weight",
		p + "attn.v.weight",
		p + "attn.o.weight",
		p + "norm2.weight",
		p + "ffn.gate.0.weight",
		p + "ffn.fc1.weight",
		p + "ffn.fc2.weight",
		p + "pos_embedding.embedding.weight",
	}
}

func encoderTensorNames(layers int) []string {
	names := []string{"token_embedding.weight"}
	for layer := range layers {
		names = append(names, encoderBlockTensorNames(layer)...)
	}
	return append(names, "norm.weight")
}

func compileEncoderBindings(metas []pytorchzip.TensorMeta, config EncoderConfig) ([]pytorchzip.TensorBinding, []int64, error) {
	names := encoderTensorNames(config.Layers)
	bindings, err := pytorchzip.CompileBindings(metas, names)
	if err != nil {
		return nil, nil, err
	}
	blockWeightBytes := make([]int64, config.Layers)
	for layer := range blockWeightBytes {
		start := tensor.SingletonExtent + layer*blockTensorCount
		for _, binding := range bindings[start : start+blockTensorCount] {
			bytes, byteErr := pytorchzip.TensorMetaBytes(binding.Meta)
			if byteErr != nil {
				return nil, nil, byteErr
			}
			blockWeightBytes[layer] += bytes
		}
	}
	return bindings, blockWeightBytes, nil
}

func (p EncoderPlan) validateRuntimeBindings() error {
	want := len(encoderTensorNames(p.Config.Layers))
	if !p.OK || !checked.PositiveInts(p.Config.Layers) || !checked.Equal(len(p.bindings), want) || !checked.Equal(len(p.blockWeightBytes), p.Config.Layers) {
		return fmt.Errorf("encoder runtime plan: ok=%t layers=%d bindings=%d/%d block_bytes=%d", p.OK, p.Config.Layers, len(p.bindings), want, len(p.blockWeightBytes))
	}
	return nil
}

type encoderBlockWeights struct {
	Norm1, Norm2 []float32
	Q, K, V, O   []float32
	Gate, FC1    []float32
	FC2          []float32
	PosEmbedding []float32
}

// EncoderStats: measured streaming behavior; PeakHeapAllocBytes is the
// largest sampled Go heap live-set DELTA over the entry baseline
// (memory-bounding evidence robust to unrelated resident heap).
type EncoderStats struct {
	Layers              int
	TokenRows           int
	EmbeddingBytesRead  int64
	BlockWeightBytes    int64
	MaxBlockWeightBytes int64
	FinalNormBytes      int64
	OutputRows          int
	OutputDim           int
	Engine              string
	EncoderWallSec      float64
	PeakHeapAllocBytes  uint64
}

// EncodeTokensStreamed runs the full encoder, loading one block's weights at
// a time (peak host memory ~ one F32-decoded block, never the checkpoint).
func EncodeTokensStreamed(checkpoint string, plan EncoderPlan, tokenIDs, mask []int) ([]float32, EncoderStats, error) {
	var stats EncoderStats
	config := plan.Config
	if len(tokenIDs) == 0 {
		return nil, stats, fmt.Errorf("encoder: no token ids")
	}
	if len(mask) != len(tokenIDs) {
		return nil, stats, fmt.Errorf("encoder: mask=%d tokens=%d", len(mask), len(tokenIDs))
	}
	if err := plan.validateRuntimeBindings(); err != nil {
		return nil, stats, err
	}
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return nil, stats, err
	}
	defer reader.Close()
	tokenBinding, ok := checked.First(plan.bindings)
	if !ok {
		return nil, stats, fmt.Errorf("encoder: token binding absent")
	}
	hidden, err := reader.ReadTensorRows(tokenBinding, tokenIDs, config.Dim)
	if err != nil {
		return nil, stats, err
	}
	stats.TokenRows = len(tokenIDs)
	stats.Engine = "host_streamed"
	started := time.Now()
	stats.EmbeddingBytesRead = int64(len(tokenIDs) * config.Dim * binaryschema.Uint16Bytes)
	buckets, err := representation.RelativePositionBuckets(len(tokenIDs), len(tokenIDs), config.RelativeBuckets, config.RelativeMaxDistance, true)
	if err != nil {
		return nil, stats, err
	}
	// Delta over the entry baseline: the streamed bound must hold regardless
	// of unrelated live heap (other tests' resident weights).
	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)
	samplePeak := func() {
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		if ms.HeapAlloc > baseline.HeapAlloc && ms.HeapAlloc-baseline.HeapAlloc > stats.PeakHeapAllocBytes {
			stats.PeakHeapAllocBytes = ms.HeapAlloc - baseline.HeapAlloc
		}
	}
	for layer := range config.Layers {
		start := tensor.SingletonExtent + layer*blockTensorCount
		weights, err := loadEncoderBlockWeights(reader, plan.bindings[start:start+blockTensorCount])
		if err != nil {
			return nil, stats, err
		}
		next, err := encoderBlockForward(hidden, mask, buckets, tensor.SingletonExtent, len(mask), config.Dim, config.Heads, config.FFNDim, config.RelativeBuckets, weights, config.NormEps, true)
		if err != nil {
			return nil, stats, err
		}
		hidden = next
		samplePeak()
		stats.Layers++
		blockBytes := plan.blockWeightBytes[layer]
		stats.BlockWeightBytes += blockBytes
		if blockBytes > stats.MaxBlockWeightBytes {
			stats.MaxBlockWeightBytes = blockBytes
		}
		runtime.GC()
	}
	normBinding, ok := checked.Last(plan.bindings)
	if !ok {
		return nil, stats, fmt.Errorf("encoder: final normalization binding is absent")
	}
	norm, err := reader.ReadBinding(normBinding)
	if err != nil {
		return nil, stats, err
	}
	hostmath.RMSNormInto(hidden, hidden, norm, len(tokenIDs), config.Dim, config.NormEps)
	dtype.RoundBF16Slice(hidden)
	stats.FinalNormBytes = int64(len(norm) * binaryschema.Uint16Bytes)
	stats.OutputRows = len(tokenIDs)
	stats.OutputDim = config.Dim
	stats.EncoderWallSec = time.Since(started).Seconds()
	return hidden, stats, nil
}

func loadEncoderBlockWeights(reader *pytorchzip.Reader, bindings []pytorchzip.TensorBinding) (encoderBlockWeights, error) {
	if len(bindings) != blockTensorCount {
		return encoderBlockWeights{}, fmt.Errorf("encoder block bindings=%d want=%d", len(bindings), blockTensorCount)
	}
	values, err := reader.ReadBindingValues(bindings)
	if err != nil {
		return encoderBlockWeights{}, err
	}
	return encoderBlockWeights{
		Norm1: values[blockNorm1], Q: values[blockQ], K: values[blockK],
		V: values[blockV], O: values[blockO], Norm2: values[blockNorm2],
		Gate: values[blockGate], FC1: values[blockFC1], FC2: values[blockFC2],
		PosEmbedding: values[blockPosition],
	}, nil
}

func validateEncoderBlockWeights(w encoderBlockWeights, dim, ffnDim, heads, numBuckets int) error {
	for _, field := range [...]struct {
		name      string
		got, want int
	}{
		{"norm1", len(w.Norm1), dim}, {"norm2", len(w.Norm2), dim},
		{"q", len(w.Q), dim * dim}, {"k", len(w.K), dim * dim},
		{"v", len(w.V), dim * dim}, {"o", len(w.O), dim * dim},
		{"gate", len(w.Gate), ffnDim * dim}, {"fc1", len(w.FC1), ffnDim * dim},
		{"fc2", len(w.FC2), dim * ffnDim}, {"pos_embedding", len(w.PosEmbedding), numBuckets * heads},
	} {
		if field.got != field.want {
			return fmt.Errorf("encoder block: %s=%d want=%d", field.name, field.got, field.want)
		}
	}
	return nil
}

// encoderBlockForward: pre-norm T5 block. bf16=true reproduces the model's
// storage dtype: f32 accumulation with a BF16 round after every op, exactly
// where the reference rounds.
func encoderBlockForward(x []float32, mask, buckets []int, batch, seq, dim, heads, ffnDim, numBuckets int, w encoderBlockWeights, eps float64, bf16 bool) ([]float32, error) {
	if !checked.PositiveInts(batch, seq, dim, heads, ffnDim, numBuckets) {
		return nil, fmt.Errorf("encoder block: bad shape batch=%d seq=%d width=%d heads=%d hidden=%d buckets=%d", batch, seq, dim, heads, ffnDim, numBuckets)
	}
	if _, ok := checked.DivExactInt(dim, heads); !ok {
		return nil, fmt.Errorf("encoder block: width=%d does not divide across heads=%d", dim, heads)
	}
	if err := checked.Length(x, batch, seq, dim); err != nil {
		return nil, fmt.Errorf("encoder block: input: %w", err)
	}
	if mask != nil {
		if err := checked.Length(mask, batch, seq); err != nil {
			return nil, fmt.Errorf("encoder block: mask: %w", err)
		}
	}
	if err := validateEncoderBlockWeights(w, dim, ffnDim, heads, numBuckets); err != nil {
		return nil, err
	}
	norm1 := rmsNormWithReduction(x, w.Norm1, batch*seq, dim, eps, bf16)
	roundBF16If(bf16, norm1)
	q := linearWithAccumulation(norm1, w.Q, batch*seq, dim, dim, bf16)
	k := linearWithAccumulation(norm1, w.K, batch*seq, dim, dim, bf16)
	v := linearWithAccumulation(norm1, w.V, batch*seq, dim, dim, bf16)
	roundBF16If(bf16, q, k, v)
	attn, err := hostmath.RelativePositionAttentionF32(q, k, v, mask, buckets, w.PosEmbedding, batch, seq, heads, dim/heads, numBuckets, bf16)
	if err != nil {
		return nil, err
	}
	proj := linearWithAccumulation(attn, w.O, batch*seq, dim, dim, bf16)
	roundBF16If(bf16, proj)
	out := append([]float32(nil), x...)
	for i := range out {
		out[i] += proj[i]
	}
	roundBF16If(bf16, out)
	norm2 := rmsNormWithReduction(out, w.Norm2, batch*seq, dim, eps, bf16)
	roundBF16If(bf16, norm2)
	gate := linearWithAccumulation(norm2, w.Gate, batch*seq, dim, ffnDim, bf16)
	roundBF16If(bf16, gate)
	hostmath.GELUTanhInPlace(gate)
	roundBF16If(bf16, gate)
	up := linearWithAccumulation(norm2, w.FC1, batch*seq, dim, ffnDim, bf16)
	roundBF16If(bf16, up)
	for i := range gate {
		gate[i] *= up[i]
	}
	roundBF16If(bf16, gate)
	ffn := linearWithAccumulation(gate, w.FC2, batch*seq, ffnDim, dim, bf16)
	roundBF16If(bf16, ffn)
	for i := range out {
		out[i] += ffn[i]
	}
	roundBF16If(bf16, out)
	return out, nil
}

func roundBF16If(enabled bool, values ...[]float32) {
	if !enabled {
		return
	}
	for _, value := range values {
		dtype.RoundBF16Slice(value)
	}
}

// linearWithAccumulation: bf16=true uses F32 accumulation (the native-dtype
// convention); bf16=false uses the F64 wider-accumulate convention.
func linearWithAccumulation(x, w []float32, rows, inDim, outDim int, bf16 bool) []float32 {
	if bf16 {
		out := make([]float32, rows*outDim)
		// hostmath.Linear accumulates F32 over the input dim in ascending
		// order — the identical per-element arithmetic, fanned out.
		hostmath.Linear(out, x, w, rows, inDim, outDim)
		return out
	}
	return hostmath.LinearF64New(x, w, nil, rows, inDim, outDim)
}

// rmsNormWithReduction: bf16=true reduces squares in F32 (native-dtype
// convention); bf16=false is the shared F64 RMSNorm.
func rmsNormWithReduction(x, weight []float32, rows, dim int, eps float64, f32Reduction bool) []float32 {
	out := make([]float32, len(x))
	if !f32Reduction {
		hostmath.RMSNormInto(out, x, weight, rows, dim, eps)
		return out
	}
	hostmath.ParallelRangeF64(rows, dim, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row, dst := x[r*dim:(r+1)*dim], out[r*dim:(r+1)*dim]
			var squares float32
			for _, value := range row {
				squares += value * value
			}
			inv := float32(1 / math.Sqrt(float64(squares/float32(dim))+eps))
			for i, value := range row {
				dst[i] = value * inv * weight[i]
			}
		}
	})
	return out
}
