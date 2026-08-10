package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// VisualSegments: contiguous non-zero-mask spans widened by the reference
// rule — a span's attention window extends one row past its end; text rows
// attend causally. Ported verbatim from the reference implementation.
func VisualSegments(modalityMask []int) [][2]int {
	separators := make([][2]int, 0)
	for i := 0; i < len(modalityMask); {
		if modalityMask[i] != 0 {
			i++
			continue
		}
		start := i
		for i < len(modalityMask) && modalityMask[i] == 0 {
			i++
		}
		end := i - 1
		if end-start+1 >= 2 {
			separators = append(separators, [2]int{start, end})
		}
	}
	segments := make([][2]int, 0, len(separators)+1)
	segStart := 0
	for _, sep := range separators {
		segEnd := sep[0] - 1
		if segEnd >= segStart {
			endExclusive := segEnd + 2
			if endExclusive > len(modalityMask) {
				endExclusive = len(modalityMask)
			}
			segments = append(segments, [2]int{segStart, endExclusive})
		}
		segStart = sep[1] + 1
	}
	if segStart < len(modalityMask) {
		segments = append(segments, [2]int{segStart, len(modalityMask)})
	}
	return segments
}

func segmentRange(segments [][2]int, tokenPos, limit int) (int, int) {
	start, end := 0, tokenPos+1
	for _, seg := range segments {
		if tokenPos >= seg[0] && tokenPos < seg[1] {
			start, end = seg[0], seg[1]
			break
		}
	}
	if start < 0 {
		start = 0
	}
	if end > limit {
		end = limit
	}
	return start, end
}

// branchRows: token indexes per branch (0 = text, nonzero = vision).
func branchRows(mask []int) (zero, nonZero []int) {
	for row, route := range mask {
		if route == 0 {
			zero = append(zero, row)
		} else {
			nonZero = append(nonZero, row)
		}
	}
	return zero, nonZero
}

// PromptLayerState: all-token intermediates of one branch-routed layer.
type PromptLayerState struct {
	cfg      Config
	mask     []int
	segments [][2]int
	// [tokens, width] flats
	NormRows            []float32
	QProj, KProj, VProj []float32
	QHeads, KHeads      []float32 // rope + per-head QK-norm applied
	Context             []float32 // attention contexts [tokens, heads*hd]
	AttnProjected       []float32 // o-proj rows [tokens, hidden]
	Output              []float32 // layer output [tokens, hidden]
	qOut, kvOut         int
}

// PromptLayerForward: one full layer over the prompt, exposing every
// intermediate the parity ladder probes. hidden is [tokens, hidden].
func PromptLayerForward(hidden []float32, mask []int, cfg Config, w LayerWeights) (*PromptLayerState, error) {
	tokens := len(mask)
	d := cfg.HiddenSize
	if tokens == 0 || len(hidden) != tokens*d {
		return nil, fmt.Errorf("routed lm prompt layer: hidden len %d != %d tokens * %d", len(hidden), tokens, d)
	}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		return nil, err
	}
	ropeInv := hostmath.RopeInvFreq(base, cfg.HeadDim)
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	s := &PromptLayerState{
		cfg: cfg, mask: mask, segments: VisualSegments(mask),
		NormRows: make([]float32, tokens*d),
		QProj:    make([]float32, tokens*qOut),
		KProj:    make([]float32, tokens*kvOut),
		VProj:    make([]float32, tokens*kvOut),
		qOut:     qOut, kvOut: kvOut,
	}
	zero, nonZero := branchRows(mask)
	// Pre-attention norm, branch-selected scale.
	for _, group := range []struct {
		rows   []int
		weight []float32
	}{{zero, w.InputNorm.Text}, {nonZero, w.InputNorm.Vision}} {
		for _, token := range group.rows {
			rmsNormRounded(s.NormRows[token*d:(token+1)*d], hidden[token*d:(token+1)*d], group.weight, 1, d, cfg.RMSNormEps)
		}
	}
	// Branch-batched Q/K/V projections (row-independent; batching == per-row).
	project := func(rows []int, q, k, v BF16Matrix) {
		if len(rows) == 0 {
			return
		}
		pack := make([]float32, len(rows)*d)
		for i, token := range rows {
			copy(pack[i*d:(i+1)*d], s.NormRows[token*d:(token+1)*d])
		}
		for _, target := range []struct {
			w   BF16Matrix
			dst []float32
			out int
		}{{q, s.QProj, qOut}, {k, s.KProj, kvOut}, {v, s.VProj, kvOut}} {
			projected := make([]float32, len(rows)*target.out)
			linearRounded(projected, pack, target.w, len(rows))
			for i, token := range rows {
				copy(target.dst[token*target.out:(token+1)*target.out], projected[i*target.out:(i+1)*target.out])
			}
		}
	}
	project(zero, w.QKV.QText, w.QKV.KText, w.QKV.VText)
	project(nonZero, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision)
	// Rope + per-head QK-norm.
	hd := cfg.HeadDim
	s.QHeads = append([]float32(nil), s.QProj...)
	s.KHeads = append([]float32(nil), s.KProj...)
	for token := 0; token < tokens; token++ {
		for head := 0; head < cfg.NumAttentionHeads; head++ {
			row := s.QHeads[token*qOut+head*hd : token*qOut+(head+1)*hd]
			applyRotaryHalfBF16(row, ropeInv, token)
			bf16RoundSlice(row)
			rmsNormRounded(row, row, w.QKV.QNorm, 1, hd, cfg.RMSNormEps)
		}
		for head := 0; head < cfg.NumKeyValueHeads; head++ {
			row := s.KHeads[token*kvOut+head*hd : token*kvOut+(head+1)*hd]
			applyRotaryHalfBF16(row, ropeInv, token)
			bf16RoundSlice(row)
			rmsNormRounded(row, row, w.QKV.KNorm, 1, hd, cfg.RMSNormEps)
		}
	}
	// Segment-windowed attention.
	s.Context = make([]float32, tokens*qOut)
	group := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	scale := 1.0 / math.Sqrt(float64(hd))
	hostmath.ParallelRangeF64(tokens, cfg.NumAttentionHeads*tokens*2*hd, func(lo, hi int) {
		scores := make([]float32, tokens)
		for tokenPos := lo; tokenPos < hi; tokenPos++ {
			start, end := segmentRange(s.segments, tokenPos, tokens)
			for head := 0; head < cfg.NumAttentionHeads; head++ {
				kvHead := head / group
				q := s.QHeads[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				window := scores[:end-start]
				for keyPos := start; keyPos < end; keyPos++ {
					k := s.KHeads[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					var dot float64
					for i := 0; i < hd; i++ {
						dot += float64(q[i]) * float64(k[i])
					}
					window[keyPos-start] = float32(dot * scale)
				}
				hostmath.SoftmaxInPlace(window)
				out := s.Context[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				for i := range out {
					out[i] = 0
				}
				for keyPos, prob := range window {
					v := s.VProj[(start+keyPos)*kvOut+kvHead*hd : (start+keyPos)*kvOut+(kvHead+1)*hd]
					for i := 0; i < hd; i++ {
						out[i] += prob * v[i]
					}
				}
				bf16RoundSlice(out)
			}
		}
	})
	// Output projection, residual, post-norm, branch MLP.
	s.AttnProjected = make([]float32, tokens*d)
	residual := make([]float32, tokens*d)
	postNorm := make([]float32, tokens*d)
	s.Output = make([]float32, tokens*d)
	inter := cfg.IntermediateSize
	body := func(rows []int, o BF16Matrix, post []float32, gate, up, down BF16Matrix) {
		if len(rows) == 0 {
			return
		}
		pack := make([]float32, len(rows)*d)
		for i, token := range rows {
			copy(pack[i*d:(i+1)*d], s.Context[token*d:(token+1)*d])
		}
		projected := make([]float32, len(rows)*d)
		linearRounded(projected, pack, o, len(rows))
		for i, token := range rows {
			copy(s.AttnProjected[token*d:(token+1)*d], projected[i*d:(i+1)*d])
			residualRow := residual[token*d : (token+1)*d]
			inputRow := hidden[token*d : (token+1)*d]
			projectedRow := projected[i*d : (i+1)*d]
			for c := 0; c < d; c++ {
				residualRow[c] = dtype.RoundBF16(inputRow[c] + projectedRow[c])
			}
			rmsNormRounded(postNorm[token*d:(token+1)*d], residualRow, post, 1, d, cfg.RMSNormEps)
		}
		groupPost := make([]float32, len(rows)*d)
		for i, token := range rows {
			copy(groupPost[i*d:(i+1)*d], postNorm[token*d:(token+1)*d])
		}
		gateValues := make([]float32, len(rows)*inter)
		upValues := make([]float32, len(rows)*inter)
		linearRounded(gateValues, groupPost, gate, len(rows))
		linearRounded(upValues, groupPost, up, len(rows))
		for i := range gateValues {
			gateValues[i] = dtype.RoundBF16(float32(silu64(float64(gateValues[i]))))
			gateValues[i] = dtype.RoundBF16(gateValues[i] * upValues[i])
		}
		downValues := make([]float32, len(rows)*d)
		linearRounded(downValues, gateValues, down, len(rows))
		for i, token := range rows {
			outRow := s.Output[token*d : (token+1)*d]
			residualRow := residual[token*d : (token+1)*d]
			downRow := downValues[i*d : (i+1)*d]
			for c := 0; c < d; c++ {
				outRow[c] = dtype.RoundBF16(residualRow[c] + downRow[c])
			}
		}
	}
	body(zero, w.QKV.OText, w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText)
	body(nonZero, w.QKV.OVision, w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision)
	return s, nil
}

// ResidentKV: per-layer prompt K/V (rope + K-norm already applied to keys).
type ResidentKV struct {
	Keys, Values []float32
	Tokens       int
	KVHeads      int
	HeadDim      int
}

// ResidentKVFromState: keys are the roped+normed KHeads, values the rounded
// V projections — identical to the reference prompt KV pack.
func (s *PromptLayerState) ResidentKV() *ResidentKV {
	return &ResidentKV{
		Keys:    append([]float32(nil), s.KHeads...),
		Values:  append([]float32(nil), s.VProj...),
		Tokens:  len(s.mask),
		KVHeads: s.cfg.NumKeyValueHeads,
		HeadDim: s.cfg.HeadDim,
	}
}

// PromptResidentKV: K/V for one layer from its input hidden rows without the
// attention/output work (the decode ladder builds per-layer caches from the
// previous layer's golden boundary).
func PromptResidentKV(hidden []float32, mask []int, cfg Config, w LayerWeights) (*ResidentKV, error) {
	tokens := len(mask)
	d := cfg.HiddenSize
	if tokens == 0 || len(hidden) != tokens*d {
		return nil, fmt.Errorf("routed lm prompt kv: hidden len %d != %d tokens * %d", len(hidden), tokens, d)
	}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		return nil, err
	}
	ropeInv := hostmath.RopeInvFreq(base, cfg.HeadDim)
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	hd := cfg.HeadDim
	kv := &ResidentKV{
		Keys:    make([]float32, tokens*kvOut),
		Values:  make([]float32, tokens*kvOut),
		Tokens:  tokens,
		KVHeads: cfg.NumKeyValueHeads,
		HeadDim: hd,
	}
	zero, nonZero := branchRows(mask)
	normRow := make([]float32, d)
	for _, group := range []struct {
		rows       []int
		normWeight []float32
		k, v       BF16Matrix
	}{
		{zero, w.InputNorm.Text, w.QKV.KText, w.QKV.VText},
		{nonZero, w.InputNorm.Vision, w.QKV.KVision, w.QKV.VVision},
	} {
		for _, token := range group.rows {
			rmsNormRounded(normRow, hidden[token*d:(token+1)*d], group.normWeight, 1, d, cfg.RMSNormEps)
			kRow := kv.Keys[token*kvOut : (token+1)*kvOut]
			vRow := kv.Values[token*kvOut : (token+1)*kvOut]
			linearRounded(kRow, normRow, group.k, 1)
			linearRounded(vRow, normRow, group.v, 1)
			for head := 0; head < cfg.NumKeyValueHeads; head++ {
				headRow := kRow[head*hd : (head+1)*hd]
				applyRotaryHalfBF16(headRow, ropeInv, token)
				bf16RoundSlice(headRow)
				rmsNormRounded(headRow, headRow, w.QKV.KNorm, 1, hd, cfg.RMSNormEps)
			}
		}
	}
	return kv, nil
}
