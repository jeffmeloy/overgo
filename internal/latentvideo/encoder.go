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
	"time"

	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
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
	tokenBinding        = 0
	leadingTensorCount  = 1
	boundaryTensorCount = 2
)

const encoderStorageDType = "BFloat16Storage"

// deriveEncoderConfig reconstructs geometry from tensor shapes; policy
// supplies only RelativeMaxDistance and NormEps.
func deriveEncoderConfig(byName map[string]pytorchzip.TensorMeta, policy EncoderConfig) (EncoderConfig, error) {
	c := EncoderConfig{RelativeMaxDistance: policy.RelativeMaxDistance, NormEps: policy.NormEps}
	norm, ok := byName["norm.weight"]
	if !ok || len(norm.Shape) != 1 || norm.Shape[0] <= 0 {
		return c, fmt.Errorf("encoder config: norm.weight missing or not 1-d")
	}
	c.Dim = int(norm.Shape[0])
	embed, ok := byName["token_embedding.weight"]
	if !ok || len(embed.Shape) != 2 || embed.Shape[1] != norm.Shape[0] || embed.Shape[0] <= 0 {
		return c, fmt.Errorf("encoder config: token_embedding.weight missing or incompatible with width=%d", c.Dim)
	}
	c.VocabSize = int(embed.Shape[0])
	pos, ok := byName["blocks.0.pos_embedding.embedding.weight"]
	if !ok || len(pos.Shape) != 2 || pos.Shape[0] <= 0 || pos.Shape[1] <= 0 {
		return c, fmt.Errorf("encoder config: blocks.0 position embedding missing or not 2-d")
	}
	c.RelativeBuckets, c.Heads = int(pos.Shape[0]), int(pos.Shape[1])
	fc1, ok := byName["blocks.0.ffn.fc1.weight"]
	if !ok || len(fc1.Shape) != 2 || fc1.Shape[1] != norm.Shape[0] || fc1.Shape[0] <= 0 {
		return c, fmt.Errorf("encoder config: blocks.0 ffn.fc1 missing or incompatible with width=%d", c.Dim)
	}
	c.FFNDim = int(fc1.Shape[0])
	for layer := 0; ; layer++ {
		if _, ok := byName[fmt.Sprintf("blocks.%d.norm1.weight", layer)]; !ok {
			c.Layers = layer
			break
		}
	}
	if c.Layers == 0 {
		return c, fmt.Errorf("encoder config: no blocks.N.norm1.weight tensors")
	}
	if c.Dim%c.Heads != 0 || c.RelativeMaxDistance <= 0 || c.NormEps <= 0 {
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
			if got.DType != encoderStorageDType {
				plan.DTypeMismatches = append(plan.DTypeMismatches, TensorIssue{Name: name, Want: encoderStorageDType, Got: got.DType})
			}
			if !int64SliceEqual(got.Shape, spec.shape) {
				plan.ShapeMismatches = append(plan.ShapeMismatches, TensorIssue{Name: name, Want: fmt.Sprint(spec.shape), Got: fmt.Sprint(got.Shape)})
			}
		}
	}
	plan.OK = len(plan.Missing) == 0 && len(plan.Unexpected) == 0 && len(plan.DTypeMismatches) == 0 && len(plan.ShapeMismatches) == 0
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
	for i := 0; i < c.Layers; i++ {
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

func int64SliceEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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

func compileEncoderBindings(metas []pytorchzip.TensorMeta, config EncoderConfig) ([]pytorchzip.TensorBinding, []int64, error) {
	names := make([]string, 0, boundaryTensorCount+config.Layers*blockTensorCount)
	names = append(names, "token_embedding.weight")
	for layer := 0; layer < config.Layers; layer++ {
		names = append(names, encoderBlockTensorNames(layer)...)
	}
	names = append(names, "norm.weight")
	bindings, err := pytorchzip.CompileBindings(metas, names)
	if err != nil {
		return nil, nil, err
	}
	blockWeightBytes := make([]int64, config.Layers)
	for layer := range blockWeightBytes {
		start := leadingTensorCount + layer*blockTensorCount
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
	want := boundaryTensorCount + p.Config.Layers*blockTensorCount
	if !p.OK || p.Config.Layers <= 0 || len(p.bindings) != want || len(p.blockWeightBytes) != p.Config.Layers {
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
// largest sampled Go heap live-set across the run (memory-bounding evidence).
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

const uint16Bytes = 2

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
	hidden, err := reader.ReadTensorRows(plan.bindings[tokenBinding], tokenIDs, config.Dim)
	if err != nil {
		return nil, stats, err
	}
	stats.TokenRows = len(tokenIDs)
	stats.Engine = "host_streamed"
	started := time.Now()
	stats.EmbeddingBytesRead = int64(len(tokenIDs) * config.Dim * uint16Bytes)
	buckets, err := compileRelativePositionBuckets(len(tokenIDs), len(tokenIDs), config.RelativeBuckets, config.RelativeMaxDistance, true)
	if err != nil {
		return nil, stats, err
	}
	samplePeak := func() {
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		if ms.HeapAlloc > stats.PeakHeapAllocBytes {
			stats.PeakHeapAllocBytes = ms.HeapAlloc
		}
	}
	for layer := 0; layer < config.Layers; layer++ {
		start := leadingTensorCount + layer*blockTensorCount
		weights, err := loadEncoderBlockWeights(reader, plan.bindings[start:start+blockTensorCount])
		if err != nil {
			return nil, stats, err
		}
		next, err := encoderBlockForward(hidden, mask, buckets, 1, len(mask), config.Dim, config.Heads, config.FFNDim, config.RelativeBuckets, weights, config.NormEps, true)
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
	norm, err := reader.ReadBinding(plan.bindings[len(plan.bindings)-1])
	if err != nil {
		return nil, stats, err
	}
	hostmath.RMSNormInto(hidden, hidden, norm, len(tokenIDs), config.Dim, config.NormEps)
	roundBF16(hidden)
	stats.FinalNormBytes = int64(len(norm) * uint16Bytes)
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
	if batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0 || ffnDim <= 0 || numBuckets <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("encoder block: bad shape batch=%d seq=%d width=%d heads=%d hidden=%d buckets=%d", batch, seq, dim, heads, ffnDim, numBuckets)
	}
	if len(x) != batch*seq*dim {
		return nil, fmt.Errorf("encoder block: input=%d want=%d", len(x), batch*seq*dim)
	}
	if mask != nil && len(mask) != batch*seq {
		return nil, fmt.Errorf("encoder block: mask=%d want=%d", len(mask), batch*seq)
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
	attn, err := relativePositionSelfAttention(q, k, v, mask, buckets, w.PosEmbedding, batch, seq, heads, dim/heads, numBuckets, bf16)
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

func roundBF16(x []float32) {
	for i, v := range x {
		x[i] = dtype.RoundBF16(v)
	}
}

func roundBF16If(enabled bool, values ...[]float32) {
	if !enabled {
		return
	}
	for _, value := range values {
		roundBF16(value)
	}
}

// linearWithAccumulation: bf16=true uses F32 accumulation (the native-dtype
// convention); bf16=false uses the F64 wider-accumulate convention.
func linearWithAccumulation(x, w []float32, rows, inDim, outDim int, bf16 bool) []float32 {
	out := make([]float32, rows*outDim)
	if bf16 {
		// hostmath.Linear accumulates F32 over the input dim in ascending
		// order — the identical per-element arithmetic, fanned out.
		hostmath.Linear(out, x, w, rows, inDim, outDim)
		return out
	}
	hostmath.ParallelRangeF64(rows, inDim*outDim, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			xr, orow := x[r*inDim:(r+1)*inDim], out[r*outDim:(r+1)*outDim]
			for o := 0; o < outDim; o++ {
				wr := w[o*inDim : (o+1)*inDim]
				var acc float64
				for i, value := range xr {
					acc += float64(value) * float64(wr[i])
				}
				orow[o] = float32(acc)
			}
		}
	})
	return out
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

func relativePositionBucket(relPos, numBuckets int, bidirectional bool, maxDist int) int {
	buckets := 0
	n := relPos
	if bidirectional {
		half := numBuckets / 2
		if n > 0 {
			buckets += half
		}
		if n < 0 {
			n = -n
		}
		numBuckets = half
	} else if n > 0 {
		n = 0
	} else {
		n = -n
	}
	maxExact := numBuckets / 2
	if n < maxExact {
		return buckets + n
	}
	if maxExact <= 0 {
		return buckets
	}
	large := maxExact + int(math.Log(float64(n)/float64(maxExact))/math.Log(float64(maxDist)/float64(maxExact))*float64(numBuckets-maxExact))
	if large > numBuckets-1 {
		large = numBuckets - 1
	}
	return buckets + large
}

func compileRelativePositionBuckets(queryRows, keyRows, numBuckets, maxDistance int, bidirectional bool) ([]int, error) {
	if queryRows <= 0 || keyRows <= 0 || numBuckets <= 0 || maxDistance <= 0 {
		return nil, fmt.Errorf("relative-position buckets: query=%d key=%d buckets=%d max_distance=%d", queryRows, keyRows, numBuckets, maxDistance)
	}
	out := make([]int, queryRows*keyRows)
	for query := 0; query < queryRows; query++ {
		for key := 0; key < keyRows; key++ {
			out[query*keyRows+key] = relativePositionBucket(key-query, numBuckets, bidirectional, maxDistance)
		}
	}
	return out, nil
}

// relativePositionSelfAttention: unscaled scores + per-bucket additive bias,
// -inf on masked keys. Per-(query,head) arithmetic is identical to the
// reference; queries fan out across workers (row-independent).
func relativePositionSelfAttention(q, k, v []float32, mask, buckets []int, embedding []float32, batch, seq, heads, headDim, numBuckets int, bf16 bool) ([]float32, error) {
	want := batch * seq * heads * headDim
	if len(q) != want || len(k) != want || len(v) != want {
		return nil, fmt.Errorf("relative-position attention: q/k/v=%d/%d/%d want=%d", len(q), len(k), len(v), want)
	}
	if len(embedding) != numBuckets*heads {
		return nil, fmt.Errorf("relative-position attention: embedding=%d want=%d", len(embedding), numBuckets*heads)
	}
	if len(buckets) != seq*seq {
		return nil, fmt.Errorf("relative-position attention: buckets=%d want=%d", len(buckets), seq*seq)
	}
	out := make([]float32, want)
	for b := 0; b < batch; b++ {
		hostmath.ParallelRangeF64(seq, heads*seq*headDim*2, func(qLo, qHi int) {
			scores := make([]float32, seq)
			for qi := qLo; qi < qHi; qi++ {
				for h := 0; h < heads; h++ {
					for ki := 0; ki < seq; ki++ {
						score := attentionDotWithAccumulation(q, k, b, qi, ki, h, seq, heads, headDim, bf16)
						bucket := buckets[qi*seq+ki]
						score += float64(embedding[bucket*heads+h])
						if mask != nil && mask[b*seq+ki] == 0 {
							score = math.Inf(-1)
						}
						scores[ki] = float32(score)
					}
					roundBF16If(bf16, scores)
					softmaxWithAccumulation(scores, bf16)
					roundBF16If(bf16, scores)
					baseOut := ((b*seq+qi)*heads + h) * headDim
					for j := 0; j < headDim; j++ {
						out[baseOut+j] = 0
					}
					for ki, prob := range scores {
						baseV := ((b*seq+ki)*heads + h) * headDim
						for j := 0; j < headDim; j++ {
							out[baseOut+j] += prob * v[baseV+j]
						}
					}
					roundBF16If(bf16, out[baseOut:baseOut+headDim])
				}
			}
		})
	}
	return out, nil
}

func softmaxWithAccumulation(row []float32, bf16 bool) {
	if !bf16 {
		hostmath.SoftmaxInPlace(row)
		return
	}
	mx := float32(math.Inf(-1))
	for _, value := range row {
		if value > mx {
			mx = value
		}
	}
	var sum float32
	for i, value := range row {
		row[i] = float32(math.Exp(float64(value - mx)))
		sum += row[i]
	}
	for i := range row {
		row[i] /= sum
	}
}

func attentionDotWithAccumulation(q, k []float32, b, qi, ki, h, seq, heads, headDim int, bf16 bool) float64 {
	qBase := ((b*seq+qi)*heads + h) * headDim
	kBase := ((b*seq+ki)*heads + h) * headDim
	if bf16 {
		var score float32
		for j := 0; j < headDim; j++ {
			score += q[qBase+j] * k[kBase+j]
		}
		return float64(score)
	}
	var score float64
	for j := 0; j < headDim; j++ {
		score += float64(q[qBase+j]) * float64(k[kBase+j])
	}
	return score
}
