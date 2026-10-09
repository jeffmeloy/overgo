// Relative-position text encoder (T5-family geometry, derived from the
// checkpoint): token embedding rows -> N pre-norm blocks (RMSNorm, relative
// -position self-attention, gated-GELU FFN) -> final RMSNorm. Weights stream
// from the PyTorch ZIP checkpoint one block at a time so peak host memory is
// bounded by a single block, never the 11+GB checkpoint. Ported behavior
// (adaptive latent_video host path), bf16-rounded at the same points.
package latentvideo

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"time"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/processmeasure"
	"overgo/internal/pytorchzip"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// encoderConfig: checkpoint geometry plus explicit profile policy
// (RelativeMaxDistance and NormEps are model facts the checkpoint does not
// carry; everything else derives from tensor shapes).
type encoderConfig struct {
	VocabSize, Layers, Dim, Heads, FFNDim, RelativeBuckets int
	RelativeMaxDistance                                    int
	NormEps                                                float64
}

// tensorIssue: one contract violation found while compiling a plan.
type tensorIssue struct {
	Name, Want, Got string
}

// encoderPlan: validated checkpoint contract with runtime bindings.
type encoderPlan struct {
	OK               bool
	Missing          []tensorIssue
	Unexpected       []tensorIssue
	DTypeMismatches  []tensorIssue
	ShapeMismatches  []tensorIssue
	Config           encoderConfig
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
func deriveEncoderConfig(byName map[string]pytorchzip.TensorMeta, policy encoderConfig) (encoderConfig, error) {
	c := encoderConfig{RelativeMaxDistance: policy.RelativeMaxDistance, NormEps: policy.NormEps}
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
func CompileEncoderPlan(metas []pytorchzip.TensorMeta, policy encoderConfig) (encoderPlan, error) {
	var plan encoderPlan
	byName := make(map[string]pytorchzip.TensorMeta, len(metas))
	for _, m := range metas {
		byName[m.Name] = m
	}
	config, configErr := deriveEncoderConfig(byName, policy)
	if configErr != nil {
		plan.Missing = append(plan.Missing, tensorIssue{Name: "config", Want: configErr.Error()})
	} else {
		plan.Config = config
		expected := expectedEncoderSpecs(config)
		for _, m := range metas {
			if _, ok := expected[m.Name]; !ok {
				plan.Unexpected = append(plan.Unexpected, tensorIssue{Name: m.Name, Got: fmt.Sprint(m.Shape)})
			}
		}
		for name, spec := range expected {
			got, ok := byName[name]
			if !ok {
				plan.Missing = append(plan.Missing, tensorIssue{Name: name, Want: fmt.Sprint(spec.shape)})
				continue
			}
			if got.DType != pytorchzip.BFloat16StorageClass() {
				plan.DTypeMismatches = append(plan.DTypeMismatches, tensorIssue{Name: name, Want: pytorchzip.BFloat16StorageClass(), Got: got.DType})
			}
			if !slices.Equal(got.Shape, spec.shape) {
				plan.ShapeMismatches = append(plan.ShapeMismatches, tensorIssue{Name: name, Want: fmt.Sprint(spec.shape), Got: fmt.Sprint(got.Shape)})
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

func expectedEncoderSpecs(c encoderConfig) map[string]tensorSpec {
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

func compileEncoderBindings(metas []pytorchzip.TensorMeta, config encoderConfig) ([]pytorchzip.TensorBinding, []int64, error) {
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

func (p encoderPlan) validateRuntimeBindings() error {
	want := len(encoderTensorNames(p.Config.Layers))
	if !p.OK || !checked.PositiveInts(p.Config.Layers) || !checked.Equal(len(p.bindings), want) || !checked.Equal(len(p.blockWeightBytes), p.Config.Layers) {
		return fmt.Errorf("encoder runtime plan: ok=%t layers=%d bindings=%d/%d block_bytes=%d", p.OK, p.Config.Layers, len(p.bindings), want, len(p.blockWeightBytes))
	}
	return nil
}

// encoderMatrix is one weight matrix of a block: widened to float32, or as the
// bfloat16 the checkpoint stores, which the streamed encoder multiplies by as
// it is.
type encoderMatrix struct {
	wide   []float32
	stored []uint16
}

func (m encoderMatrix) elements() int { return max(len(m.wide), len(m.stored)) }

type encoderBlockWeights struct {
	Norm1, Norm2 []float32
	Q, K, V, O   encoderMatrix
	Gate, FC1    encoderMatrix
	FC2          encoderMatrix
	PosEmbedding []float32
}

// blockMatrices are the places, among a block's tensors, of its weight
// matrices, in the order a pass keeps their storage.
var blockMatrices = [...]int{blockQ, blockK, blockV, blockO, blockGate, blockFC1, blockFC2}

// encoderStats: measured streaming behavior; PeakHeapAllocBytes is the
// largest sampled Go heap live-set DELTA over the entry baseline
// (memory-bounding evidence robust to unrelated resident heap).
type encoderStats struct {
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

// encoderSequence is one token sequence and the mask of the rows it keeps.
type encoderSequence struct {
	tokens, mask []int
}

// EncodeTokensStreamed runs the full encoder, loading one block's weights at
// a time (peak host memory ~ one block as stored, never the checkpoint).
// It encodes the rows up to the last one the mask keeps and returns those:
// a row the mask keeps attends to no row it leaves out, so the padding after
// a prompt enters none of the prompt's rows, and a prompt of twenty tokens is
// twenty rows of work where its padded sequence is 512.
func EncodeTokensStreamed(ctx context.Context, checkpoint string, plan encoderPlan, tokenIDs, mask []int) ([]float32, encoderStats, error) {
	if err := ctx.Err(); err != nil {
		return nil, encoderStats{}, err
	}
	encoded, stats, err := encodeSequencesStreamed(ctx, checkpoint, plan, []encoderSequence{{tokens: tokenIDs, mask: mask}})
	if err != nil {
		return nil, stats, err
	}
	return encoded[0], stats, nil
}

// encodeSequencesStreamed runs the encoder over several sequences in one pass
// over the checkpoint: each block's weights are read once, as stored, into
// storage the pass keeps from block to block, and every sequence goes through
// them. Reading and widening the checkpoint for one prompt took 3.3 seconds
// whatever the prompt's length, and a request that carries two prompts paid it
// twice.
func encodeSequencesStreamed(ctx context.Context, checkpoint string, plan encoderPlan, sequences []encoderSequence) ([][]float32, encoderStats, error) {
	var stats encoderStats
	config := plan.Config
	sequences = slices.Clone(sequences)
	for index, sequence := range sequences {
		if len(sequence.mask) != len(sequence.tokens) {
			return nil, stats, fmt.Errorf("encoder: mask=%d tokens=%d", len(sequence.mask), len(sequence.tokens))
		}
		for len(sequence.mask) > 0 && sequence.mask[len(sequence.mask)-1] == 0 {
			sequence.mask = sequence.mask[:len(sequence.mask)-1]
		}
		sequence.tokens = sequence.tokens[:len(sequence.mask)]
		if len(sequence.tokens) == 0 {
			return nil, stats, fmt.Errorf("encoder: no token ids")
		}
		sequences[index] = sequence
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
	hidden, buckets := make([][]float32, len(sequences)), make([][]int, len(sequences))
	for index, sequence := range sequences {
		if hidden[index], err = reader.ReadTensorRows(tokenBinding, sequence.tokens, config.Dim); err != nil {
			return nil, stats, err
		}
		rows := len(sequence.tokens)
		stats.TokenRows += rows
		if buckets[index], err = representation.RelativePositionBuckets(rows, rows, config.RelativeBuckets, config.RelativeMaxDistance, true); err != nil {
			return nil, stats, err
		}
	}
	stats.Engine = "host_streamed"
	started := time.Now()
	stats.EmbeddingBytesRead = int64(stats.TokenRows * config.Dim * binaryschema.Uint16Bytes)
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
	var kept [len(blockMatrices)][]uint16
	for layer := range config.Layers {
		if err := ctx.Err(); err != nil {
			return nil, stats, err
		}
		start := tensor.SingletonExtent + layer*blockTensorCount
		read := processmeasure.Stage(ctx, "text encoder weights read")
		weights, err := loadEncoderBlockWeights(reader, plan.bindings[start:start+blockTensorCount], &kept)
		read()
		if err != nil {
			return nil, stats, err
		}
		for index, sequence := range sequences {
			if err := ctx.Err(); err != nil {
				return nil, stats, err
			}
			hidden[index], err = encoderBlockForward(hidden[index], sequence.mask, buckets[index], tensor.SingletonExtent, len(sequence.mask), config.Dim, config.Heads, config.FFNDim, config.RelativeBuckets, weights, config.NormEps, true)
			if err != nil {
				return nil, stats, err
			}
		}
		samplePeak()
		stats.Layers++
		blockBytes := plan.blockWeightBytes[layer]
		stats.BlockWeightBytes += blockBytes
		if blockBytes > stats.MaxBlockWeightBytes {
			stats.MaxBlockWeightBytes = blockBytes
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, stats, err
	}
	normBinding, ok := checked.Last(plan.bindings)
	if !ok {
		return nil, stats, fmt.Errorf("encoder: final normalization binding is absent")
	}
	norm, err := reader.ReadBinding(normBinding)
	if err != nil {
		return nil, stats, err
	}
	for index, sequence := range sequences {
		hostmath.RMSNormInto(hidden[index], hidden[index], norm, len(sequence.tokens), config.Dim, config.NormEps)
		dtype.RoundBF16Slice(hidden[index])
	}
	stats.FinalNormBytes = int64(len(norm) * binaryschema.Uint16Bytes)
	stats.OutputRows = stats.TokenRows
	stats.OutputDim = config.Dim
	stats.EncoderWallSec = time.Since(started).Seconds()
	return hidden, stats, nil
}

// loadEncoderBlockWeights reads one block: its matrices as stored, into the
// storage kept from the block before, and its three small tensors widened.
func loadEncoderBlockWeights(reader *pytorchzip.Reader, bindings []pytorchzip.TensorBinding, kept *[len(blockMatrices)][]uint16) (encoderBlockWeights, error) {
	if len(bindings) != blockTensorCount {
		return encoderBlockWeights{}, fmt.Errorf("encoder block bindings=%d want=%d", len(bindings), blockTensorCount)
	}
	var err error
	for index, place := range blockMatrices {
		if kept[index], err = reader.ReadBF16(bindings[place], kept[index]); err != nil {
			return encoderBlockWeights{}, err
		}
	}
	var small [blockTensorCount][]float32
	for _, place := range [...]int{blockNorm1, blockNorm2, blockPosition} {
		if small[place], err = reader.ReadBinding(bindings[place]); err != nil {
			return encoderBlockWeights{}, err
		}
	}
	weights := encoderBlockWeights{Norm1: small[blockNorm1], Norm2: small[blockNorm2], PosEmbedding: small[blockPosition]}
	// In blockMatrices' order.
	for index, matrix := range [...]*encoderMatrix{&weights.Q, &weights.K, &weights.V, &weights.O, &weights.Gate, &weights.FC1, &weights.FC2} {
		matrix.stored = kept[index]
	}
	return weights, nil
}

func validateEncoderBlockWeights(w encoderBlockWeights, dim, ffnDim, heads, numBuckets int, bf16 bool) error {
	for _, field := range [...]struct {
		name      string
		got, want int
	}{
		{"norm1", len(w.Norm1), dim}, {"norm2", len(w.Norm2), dim},
		{"q", w.Q.elements(), dim * dim}, {"k", w.K.elements(), dim * dim},
		{"v", w.V.elements(), dim * dim}, {"o", w.O.elements(), dim * dim},
		{"gate", w.Gate.elements(), ffnDim * dim}, {"fc1", w.FC1.elements(), ffnDim * dim},
		{"fc2", w.FC2.elements(), dim * ffnDim}, {"pos_embedding", len(w.PosEmbedding), numBuckets * heads},
	} {
		if field.got != field.want {
			return fmt.Errorf("encoder block: %s=%d want=%d", field.name, field.got, field.want)
		}
	}
	for _, matrix := range [...]encoderMatrix{w.Q, w.K, w.V, w.O, w.Gate, w.FC1, w.FC2} {
		if matrix.stored != nil && !bf16 {
			return fmt.Errorf("encoder block: a matrix as stored is multiplied in its own arithmetic, not the wider one")
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
	if err := validateEncoderBlockWeights(w, dim, ffnDim, heads, numBuckets, bf16); err != nil {
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
// convention); bf16=false uses the F64 wider-accumulate convention. A matrix
// as stored is widened where it is multiplied, to the value it has widened
// beforehand.
func linearWithAccumulation(x []float32, w encoderMatrix, rows, inDim, outDim int, bf16 bool) []float32 {
	if bf16 {
		out := make([]float32, rows*outDim)
		// hostmath.Linear accumulates F32 over the input dim in ascending
		// order — the identical per-element arithmetic, fanned out.
		if w.stored != nil {
			hostmath.LinearBF16(out, x, w.stored, rows, inDim, outDim)
		} else {
			hostmath.Linear(out, x, w.wide, rows, inDim, outDim)
		}
		return out
	}
	return hostmath.LinearF64New(x, w.wide, nil, rows, inDim, outDim)
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
