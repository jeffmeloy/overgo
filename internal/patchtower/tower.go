package patchtower

import (
	"fmt"
	"io"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// Checkpoint tensor names (safetensors index keys of the reference model).
const (
	PatchEmbedWeightName = "model.visual.vision_tower.patch_embed.proj.weight"
	PatchEmbedBiasName   = "model.visual.vision_tower.patch_embed.proj.bias"
	PosEmbedName         = "model.visual.vision_tower.pos_embed"
	blockPrefixFormat    = "model.visual.vision_tower.blocks.%d."
	mergerProj1Weight    = "model.visual.merger.proj1.weight"
	mergerProj1Bias      = "model.visual.merger.proj1.bias"
	mergerProj2Weight    = "model.visual.merger.proj2.weight"
	mergerProj2Bias      = "model.visual.merger.proj2.bias"
	mergerPool0Weight    = "model.visual.merger.pooler.predictor.0.weight"
	mergerPool0Bias      = "model.visual.merger.pooler.predictor.0.bias"
	mergerPool2Weight    = "model.visual.merger.pooler.predictor.2.weight"
	mergerPool2Bias      = "model.visual.merger.pooler.predictor.2.bias"
)

// MaterializeF32: reads a named tensor as f32, verifying element count when
// wantElements >= 0.
func MaterializeF32(src *safetensors.Source, name string, wantElements int) ([]float32, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("patch tower: missing tensor %s", name)
	}
	elements := int(t.Elements())
	if wantElements >= 0 && elements != wantElements {
		return nil, fmt.Errorf("patch tower: tensor %s elements %d != %d (shape %v)", name, elements, wantElements, t.Shape)
	}
	reader, err := safetensors.F32Reader(t)
	if err != nil {
		return nil, fmt.Errorf("patch tower: %s: %w", name, err)
	}
	raw := make([]byte, elements*4)
	if _, err := io.ReadFull(reader, raw); err != nil {
		return nil, fmt.Errorf("patch tower: %s read: %w", name, err)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out, nil
}

type PatchEmbedWeights struct {
	Weight []float32 // [hidden, 3, patch, patch]
	Bias   []float32 // [hidden]
}

type PosEmbedTable struct {
	Table  []float32 // [side*side, hidden]
	Side   int
	Hidden int
}

type BlockWeights struct {
	Norm1Weight, Norm1Bias []float32
	QKVWeight, QKVBias     []float32 // [3h, h], [3h]
	ProjWeight, ProjBias   []float32 // [h, h], [h]
	Norm2Weight, Norm2Bias []float32
	FC1Weight, FC1Bias     []float32 // [inter, h], [inter]
	FC2Weight, FC2Bias     []float32 // [h, inter], [h]
}

type MergerWeights struct {
	Proj1Weight, Proj1Bias []float32 // [out, h], [out]
	Proj2Weight, Proj2Bias []float32 // [out, out], [out]
	Pool0Weight, Pool0Bias []float32 // [out, 2*out], [out]
	Pool2Weight, Pool2Bias []float32 // [out, out], [out]
}

func LoadPatchEmbedWeights(src *safetensors.Source, spec Spec) (PatchEmbedWeights, error) {
	weight, err := MaterializeF32(src, PatchEmbedWeightName, spec.Hidden*RGBChannels*spec.PatchSize*spec.PatchSize)
	if err != nil {
		return PatchEmbedWeights{}, err
	}
	bias, err := MaterializeF32(src, PatchEmbedBiasName, spec.Hidden)
	if err != nil {
		return PatchEmbedWeights{}, err
	}
	return PatchEmbedWeights{Weight: weight, Bias: bias}, nil
}

func LoadPosEmbedTable(src *safetensors.Source, spec Spec) (PosEmbedTable, error) {
	table, err := MaterializeF32(src, PosEmbedName, spec.PositionSide*spec.PositionSide*spec.Hidden)
	if err != nil {
		return PosEmbedTable{}, err
	}
	return PosEmbedTable{Table: table, Side: spec.PositionSide, Hidden: spec.Hidden}, nil
}

func LoadBlockWeights(src *safetensors.Source, spec Spec, layer int) (BlockWeights, error) {
	if layer < 0 || layer >= spec.Depth {
		return BlockWeights{}, fmt.Errorf("patch tower block: layer=%d outside depth=%d", layer, spec.Depth)
	}
	prefix := fmt.Sprintf(blockPrefixFormat, layer)
	h, f := spec.Hidden, spec.BlockMLPInter
	var w BlockWeights
	var err error
	load := func(dst *[]float32, name string, elements int) {
		if err != nil {
			return
		}
		*dst, err = MaterializeF32(src, prefix+name, elements)
	}
	load(&w.Norm1Weight, "norm1.weight", h)
	load(&w.Norm1Bias, "norm1.bias", h)
	load(&w.QKVWeight, "attn.qkv.weight", 3*h*h)
	load(&w.QKVBias, "attn.qkv.bias", 3*h)
	load(&w.ProjWeight, "attn.proj.weight", h*h)
	load(&w.ProjBias, "attn.proj.bias", h)
	load(&w.Norm2Weight, "norm2.weight", h)
	load(&w.Norm2Bias, "norm2.bias", h)
	load(&w.FC1Weight, "mlp.fc1.weight", f*h)
	load(&w.FC1Bias, "mlp.fc1.bias", f)
	load(&w.FC2Weight, "mlp.fc2.weight", h*f)
	load(&w.FC2Bias, "mlp.fc2.bias", h)
	if err != nil {
		return BlockWeights{}, fmt.Errorf("patch tower block %d: %w", layer, err)
	}
	return w, nil
}

func LoadMergerWeights(src *safetensors.Source, spec Spec) (MergerWeights, error) {
	h, out := spec.Hidden, spec.OutHidden
	var w MergerWeights
	var err error
	load := func(dst *[]float32, name string, elements int) {
		if err != nil {
			return
		}
		*dst, err = MaterializeF32(src, name, elements)
	}
	load(&w.Proj1Weight, mergerProj1Weight, out*h)
	load(&w.Proj1Bias, mergerProj1Bias, out)
	load(&w.Proj2Weight, mergerProj2Weight, out*out)
	load(&w.Proj2Bias, mergerProj2Bias, out)
	load(&w.Pool0Weight, mergerPool0Weight, out*2*out)
	load(&w.Pool0Bias, mergerPool0Bias, out)
	load(&w.Pool2Weight, mergerPool2Weight, out*out)
	load(&w.Pool2Bias, mergerPool2Bias, out)
	if err != nil {
		return MergerWeights{}, fmt.Errorf("patch tower merger: %w", err)
	}
	return w, nil
}

// addBiasBF16: x = bf16(x + bias[col]) per element (runtime_bias_bf16).
func addBiasBF16(x, bias []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := x[r*cols : (r+1)*cols]
		for c, v := range row {
			row[c] = dtype.RoundBF16(v + bias[c])
		}
	}
}

// patchEmbedSourcePatch: undoes the merged-block patch order back to the
// row-major token index used by the position table.
func patchEmbedSourcePatch(rowMajorPatch, gridH, gridW, merge int) (int, error) {
	spatial := gridH * gridW
	if rowMajorPatch < 0 {
		return 0, fmt.Errorf("patch tower patch_embed source: negative patch %d", rowMajorPatch)
	}
	t, spatialPatch := rowMajorPatch/spatial, rowMajorPatch%spatial
	if spatialPatch < 0 || spatialPatch >= spatial {
		return 0, fmt.Errorf("patch tower patch_embed source: patch %d outside grid %dx%d", rowMajorPatch, gridH, gridW)
	}
	row, col := spatialPatch/gridW, spatialPatch%gridW
	blockW := gridW / merge
	return t*spatial + (row/merge)*blockW*merge*merge + (col/merge)*merge*merge + (row%merge)*merge + (col % merge), nil
}

func patchEmbedOne(pixelValues []float32, srcPatch, channel int, spec Spec, w PatchEmbedWeights) float32 {
	patchArea := spec.PatchSize * spec.PatchSize
	pvBase := srcPatch * spec.PatchDim
	wBase := channel * RGBChannels * patchArea
	acc := float64(w.Bias[channel])
	for c := 0; c < RGBChannels; c++ {
		for p := 0; p < patchArea; p++ {
			x := dtype.RoundBF16(pixelValues[pvBase+c*patchArea+p])
			acc += float64(x) * float64(w.Weight[wBase+c*patchArea+p])
		}
	}
	return dtype.RoundBF16(float32(acc))
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// samplePosEmbed: bilinear interpolation of the square position table onto
// the actual grid, f64 mix, bf16-rounded sample.
func samplePosEmbed(pos PosEmbedTable, token, gridH, gridW, channel int) float32 {
	spatial := gridH * gridW
	spatialToken := token % spatial
	row, col := spatialToken/gridW, spatialToken%gridW
	y := float64(pos.Side)*(float64(row)+0.5)/float64(gridH) - 0.5
	x := float64(pos.Side)*(float64(col)+0.5)/float64(gridW) - 0.5
	y0, x0 := int(math.Floor(y)), int(math.Floor(x))
	y1, x1 := y0+1, x0+1
	y0, y1 = clampInt(y0, 0, pos.Side-1), clampInt(y1, 0, pos.Side-1)
	x0, x1 = clampInt(x0, 0, pos.Side-1), clampInt(x1, 0, pos.Side-1)
	wy, wx := y-float64(y0), x-float64(x0)
	v00 := float64(pos.Table[(y0*pos.Side+x0)*pos.Hidden+channel])
	v01 := float64(pos.Table[(y0*pos.Side+x1)*pos.Hidden+channel])
	v10 := float64(pos.Table[(y1*pos.Side+x0)*pos.Hidden+channel])
	v11 := float64(pos.Table[(y1*pos.Side+x1)*pos.Hidden+channel])
	top := v00*(1-wx) + v01*wx
	bottom := v10*(1-wx) + v11*wx
	return dtype.RoundBF16(float32(top*(1-wy) + bottom*wy))
}

// PreBlock0Values: conv patch embed + interpolated pos-embed, bf16 rounded.
func PreBlock0Values(pixelValues []float32, gridT, gridH, gridW int, spec Spec, weights PatchEmbedWeights, pos PosEmbedTable) ([]float32, error) {
	if gridT <= 0 || gridH <= 0 || gridW <= 0 {
		return nil, fmt.Errorf("patch tower pre_block0: invalid grid [%d,%d,%d]", gridT, gridH, gridW)
	}
	if gridH%spec.MergeSize != 0 || gridW%spec.MergeSize != 0 {
		return nil, fmt.Errorf("patch tower pre_block0: grid [%d,%d] not divisible by merge=%d", gridH, gridW, spec.MergeSize)
	}
	nPatch := gridT * gridH * gridW
	if len(pixelValues) != nPatch*spec.PatchDim {
		return nil, fmt.Errorf("patch tower pre_block0: pixel_values len %d != %d", len(pixelValues), nPatch*spec.PatchDim)
	}
	if pos.Hidden != spec.Hidden || len(pos.Table) != pos.Side*pos.Side*pos.Hidden {
		return nil, fmt.Errorf("patch tower pre_block0: pos table len %d side %d hidden %d", len(pos.Table), pos.Side, pos.Hidden)
	}
	out := make([]float32, nPatch*spec.Hidden)
	var firstErr error
	hostmath.ParallelRangeF64(nPatch, spec.Hidden*RGBChannels*spec.PatchSize*spec.PatchSize, func(lo, hi int) {
		for token := lo; token < hi; token++ {
			srcPatch, err := patchEmbedSourcePatch(token, gridH, gridW, spec.MergeSize)
			if err != nil {
				firstErr = err
				return
			}
			dst := out[token*spec.Hidden : (token+1)*spec.Hidden]
			for channel := 0; channel < spec.Hidden; channel++ {
				patch := patchEmbedOne(pixelValues, srcPatch, channel, spec, weights)
				position := samplePosEmbed(pos, token, gridH, gridW, channel)
				dst[channel] = dtype.RoundBF16(patch + position)
			}
		}
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// fullAttention: bidirectional softmax attention over [rows, heads, hd] flat
// q/k/v with score scale 1/sqrt(hd). Reference discipline (the verified
// block plan's softmax kernel): f32 scores from exact products, f32 exp with
// f64 sum, probabilities bf16-rounded before the value mix.
func fullAttention(out, q, k, v []float32, rows, heads, hd int) {
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	hostmath.ParallelRangeF64(rows*heads, 2*rows*hd, func(lo, hi int) {
		scores := make([]float32, rows)
		for job := lo; job < hi; job++ {
			qi, h := job/heads, job%heads
			qRow := q[(qi*heads+h)*hd : (qi*heads+h+1)*hd]
			maxScore := float32(math.Inf(-1))
			for ki := 0; ki < rows; ki++ {
				kRow := k[(ki*heads+h)*hd : (ki*heads+h+1)*hd]
				var dot float64
				for d := 0; d < hd; d++ {
					dot += float64(qRow[d]) * float64(kRow[d])
				}
				score := float32(dot) * scale
				scores[ki] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var sum float64
			for ki, score := range scores {
				e := float32(math.Exp(float64(score - maxScore)))
				scores[ki] = e
				sum += float64(e)
			}
			inv := float32(1.0 / sum)
			outRow := out[(qi*heads+h)*hd : (qi*heads+h+1)*hd]
			for d := range outRow {
				outRow[d] = 0
			}
			for ki := 0; ki < rows; ki++ {
				prob := dtype.RoundBF16(scores[ki] * inv)
				if prob == 0 {
					continue
				}
				vRow := v[(ki*heads+h)*hd : (ki*heads+h+1)*hd]
				for d := 0; d < hd; d++ {
					outRow[d] += prob * vRow[d]
				}
			}
		}
	})
}

// BlockStageOutputs: intermediates for parity probing.
type BlockStageOutputs struct {
	Norm1         []float32 // [rows, hidden] LayerNorm output, bf16-rounded
	QKV           []float32 // [rows, 3*hidden] post-bias
	AttnProjected []float32 // [rows, hidden] o-proj + bias
	Output        []float32 // [rows, hidden] block output
}

// BlockForwardStages: one pre-norm block, host port of the reference block
// plan op order (LayerNorm -> round -> fused QKV -> +bias(bf16) -> split ->
// full attention -> o-proj -> +bias(bf16) -> residual(bf16) -> LayerNorm ->
// round -> fc1 -> erf-GELU(+bias, f64, bf16) -> fc2 -> +bias(bf16) ->
// residual add(bf16)).
func BlockForwardStages(pre []float32, rows int, spec Spec, w BlockWeights) (BlockStageOutputs, error) {
	h, inter, heads := spec.Hidden, spec.BlockMLPInter, spec.Heads
	if len(pre) != rows*h {
		return BlockStageOutputs{}, fmt.Errorf("patch tower block: pre len %d != %d", len(pre), rows*h)
	}
	hd := h / heads
	norm1 := make([]float32, rows*h)
	hostmath.LayerNormInto(norm1, pre, w.Norm1Weight, w.Norm1Bias, rows, h, LayerNormEps)
	dtype.RoundBF16Slice(norm1)
	qkv := make([]float32, rows*3*h)
	hostmath.LinearF64(qkv, norm1, w.QKVWeight, nil, rows, h, 3*h)
	addBiasBF16(qkv, w.QKVBias, rows, 3*h)
	q := make([]float32, rows*h)
	k := make([]float32, rows*h)
	v := make([]float32, rows*h)
	for token := 0; token < rows; token++ {
		src := token * 3 * h
		dst := token * h
		copy(q[dst:dst+h], qkv[src:src+h])
		copy(k[dst:dst+h], qkv[src+h:src+2*h])
		copy(v[dst:dst+h], qkv[src+2*h:src+3*h])
	}
	attn := make([]float32, rows*h)
	fullAttention(attn, q, k, v, rows, heads, hd)
	projected := q // reuse
	hostmath.LinearF64(projected, attn, w.ProjWeight, nil, rows, h, h)
	addBiasBF16(projected, w.ProjBias, rows, h)
	resid := k // reuse
	for i := range resid {
		resid[i] = dtype.RoundBF16(dtype.RoundBF16(pre[i]) + projected[i])
	}
	normed := v // reuse
	hostmath.LayerNormInto(normed, resid, w.Norm2Weight, w.Norm2Bias, rows, h, LayerNormEps)
	dtype.RoundBF16Slice(normed)
	fc1 := make([]float32, rows*inter)
	hostmath.LinearF64(fc1, normed, w.FC1Weight, nil, rows, h, inter)
	for r := 0; r < rows; r++ {
		row := fc1[r*inter : (r+1)*inter]
		for c, x := range row {
			value := float64(x + w.FC1Bias[c])
			row[c] = dtype.RoundBF16(float32(0.5 * value * (1.0 + math.Erf(value/math.Sqrt2))))
		}
	}
	fc2 := normed // reuse
	hostmath.LinearF64(fc2, fc1, w.FC2Weight, nil, rows, inter, h)
	addBiasBF16(fc2, w.FC2Bias, rows, h)
	out := make([]float32, rows*h)
	for i := range out {
		out[i] = dtype.RoundBF16(resid[i] + fc2[i])
	}
	return BlockStageOutputs{Norm1: norm1, QKV: qkv, AttnProjected: projected, Output: out}, nil
}

// MergerScratch: per-row merger workspace.
type MergerScratch struct {
	projected, pooled, scores []float32
	fused, hidden, weighted   []float32
	score                     []float32
}

func NewMergerScratch(spec Spec) MergerScratch {
	group := spec.MergeSize * spec.MergeSize
	return MergerScratch{
		projected: make([]float32, group*spec.OutHidden),
		pooled:    make([]float32, spec.OutHidden),
		scores:    make([]float32, group*spec.OutHidden),
		fused:     make([]float32, 2*spec.OutHidden),
		hidden:    make([]float32, spec.OutHidden),
		weighted:  make([]float32, spec.OutHidden),
		score:     make([]float32, group),
	}
}

// ValidateMergerInputs: returns merged image-feature row count.
func ValidateMergerInputs(blockLast []float32, gridT, gridH, gridW int, spec Spec) (int, error) {
	if gridT <= 0 || gridH <= 0 || gridW <= 0 {
		return 0, fmt.Errorf("patch tower merger: invalid grid [%d,%d,%d]", gridT, gridH, gridW)
	}
	if spec.MergeSize != 2 {
		return 0, fmt.Errorf("patch tower merger: merge=%d, only 2x pooler is implemented", spec.MergeSize)
	}
	if gridH%spec.MergeSize != 0 || gridW%spec.MergeSize != 0 {
		return 0, fmt.Errorf("patch tower merger: grid [%d,%d] not divisible by merge=%d", gridH, gridW, spec.MergeSize)
	}
	nPatch := gridT * gridH * gridW
	if len(blockLast) != nPatch*spec.Hidden {
		return 0, fmt.Errorf("patch tower merger: block_last len %d != %d", len(blockLast), nPatch*spec.Hidden)
	}
	return gridT * (gridH / spec.MergeSize) * (gridW / spec.MergeSize), nil
}

// MergerRowInto: one merged image-feature row (proj1 per member, mean pool,
// score MLP with erf-GELU, per-channel softmax mix, erf-GELU, proj2).
func MergerRowInto(dst, blockLast []float32, outRow, gridH, gridW int, spec Spec, merger MergerWeights, scratch *MergerScratch) {
	outH, outW := gridH/spec.MergeSize, gridW/spec.MergeSize
	outSpatial := outH * outW
	t := outRow / outSpatial
	spatial := outRow % outSpatial
	oh, ow := spatial/outW, spatial%outW

	projected := scratch.projected
	for dy := 0; dy < spec.MergeSize; dy++ {
		for dx := 0; dx < spec.MergeSize; dx++ {
			pos := dy*spec.MergeSize + dx
			src := t*gridH*gridW + (oh*spec.MergeSize+dy)*gridW + (ow*spec.MergeSize + dx)
			srcRow := blockLast[src*spec.Hidden : (src+1)*spec.Hidden]
			member := projected[pos*spec.OutHidden : (pos+1)*spec.OutHidden]
			hostmath.LinearF64(member, srcRow, merger.Proj1Weight, merger.Proj1Bias, 1, spec.Hidden, spec.OutHidden)
			dtype.RoundBF16Slice(member)
		}
	}

	pooled := scratch.pooled
	group := spec.MergeSize * spec.MergeSize
	for c := 0; c < spec.OutHidden; c++ {
		var acc float32
		for pos := 0; pos < group; pos++ {
			acc += projected[pos*spec.OutHidden+c]
		}
		pooled[c] = dtype.RoundBF16(acc / float32(group))
	}

	scores := scratch.scores
	fused := scratch.fused
	hidden := scratch.hidden
	for pos := 0; pos < group; pos++ {
		copy(fused[:spec.OutHidden], projected[pos*spec.OutHidden:(pos+1)*spec.OutHidden])
		copy(fused[spec.OutHidden:], pooled)
		hostmath.LinearF64(hidden, fused, merger.Pool0Weight, merger.Pool0Bias, 1, 2*spec.OutHidden, spec.OutHidden)
		dtype.RoundBF16Slice(hidden)
		for i, v := range hidden {
			hidden[i] = dtype.RoundBF16(float32(hostmath.GELUErf(float64(v))))
		}
		score := scores[pos*spec.OutHidden : (pos+1)*spec.OutHidden]
		hostmath.LinearF64(score, hidden, merger.Pool2Weight, merger.Pool2Bias, 1, spec.OutHidden, spec.OutHidden)
		dtype.RoundBF16Slice(score)
	}

	weighted := scratch.weighted
	score := scratch.score
	for c := 0; c < spec.OutHidden; c++ {
		for pos := 0; pos < group; pos++ {
			score[pos] = scores[pos*spec.OutHidden+c]
		}
		hostmath.SoftmaxInPlace(score)
		var acc float64
		for pos, prob := range score {
			acc += float64(projected[pos*spec.OutHidden+c]) * float64(prob)
		}
		weighted[c] = dtype.RoundBF16(float32(acc))
	}
	for i, v := range weighted {
		weighted[i] = dtype.RoundBF16(float32(hostmath.GELUErf(float64(v))))
	}

	hostmath.LinearF64(dst, weighted, merger.Proj2Weight, merger.Proj2Bias, 1, spec.OutHidden, spec.OutHidden)
	dtype.RoundBF16Slice(dst)
}
