package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// DecodeRow: one decode token through one layer against a resident cache.
// decodeMask has len resident.Tokens+1 (the new token's modality at the
// end); segments are the PROMPT visual segments. Appends the token's K/V to
// the resident cache and returns the layer output row.
func DecodeRow(tokenRow []float32, decodeMask []int, segments [][2]int, resident *ResidentKV, cfg Config, w LayerWeights) ([]float32, error) {
	d := cfg.HiddenSize
	if len(tokenRow) != d {
		return nil, fmt.Errorf("routed lm decode: token row len %d != hidden %d", len(tokenRow), d)
	}
	if resident == nil {
		return nil, fmt.Errorf("routed lm decode: nil resident kv cache")
	}
	tokenPos := resident.Tokens
	if len(decodeMask) != tokenPos+1 {
		return nil, fmt.Errorf("routed lm decode: decode mask len %d != token pos+1 %d", len(decodeMask), tokenPos+1)
	}
	if resident.KVHeads != cfg.NumKeyValueHeads || resident.HeadDim != cfg.HeadDim {
		return nil, fmt.Errorf("routed lm decode: resident shape heads=%d dim=%d, config heads=%d dim=%d", resident.KVHeads, resident.HeadDim, cfg.NumKeyValueHeads, cfg.HeadDim)
	}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		return nil, err
	}
	ropeInv := hostmath.RopeInvFreq(base, cfg.HeadDim)
	modality := decodeMask[tokenPos]
	normWeight := w.InputNorm.Text
	qW, kW, vW, oW := w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText
	postW, gateW, upW, downW := w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText
	if modality > 0 {
		normWeight = w.InputNorm.Vision
		qW, kW, vW, oW = w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision
		postW, gateW, upW, downW = w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision
	}
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	hd := cfg.HeadDim

	normed := make([]float32, d)
	rmsNormRounded(normed, tokenRow, normWeight, 1, d, cfg.RMSNormEps)
	q := make([]float32, qOut)
	k := make([]float32, kvOut)
	v := make([]float32, kvOut)
	linearRounded(q, normed, qW, 1)
	linearRounded(k, normed, kW, 1)
	linearRounded(v, normed, vW, 1)
	for head := 0; head < cfg.NumAttentionHeads; head++ {
		row := q[head*hd : (head+1)*hd]
		applyRotaryHalfBF16(row, ropeInv, tokenPos)
		bf16RoundSlice(row)
		rmsNormRounded(row, row, w.QKV.QNorm, 1, hd, cfg.RMSNormEps)
	}
	for head := 0; head < cfg.NumKeyValueHeads; head++ {
		row := k[head*hd : (head+1)*hd]
		applyRotaryHalfBF16(row, ropeInv, tokenPos)
		bf16RoundSlice(row)
		rmsNormRounded(row, row, w.QKV.KNorm, 1, hd, cfg.RMSNormEps)
	}
	resident.Keys = append(resident.Keys, k...)
	resident.Values = append(resident.Values, v...)
	resident.Tokens++

	// Attention over the resident window.
	start, end := segmentRange(segments, tokenPos, resident.Tokens)
	group := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	scale := 1.0 / math.Sqrt(float64(hd))
	context := make([]float32, qOut)
	scores := make([]float32, end-start)
	for head := 0; head < cfg.NumAttentionHeads; head++ {
		kvHead := head / group
		qRow := q[head*hd : (head+1)*hd]
		for keyPos := start; keyPos < end; keyPos++ {
			kRow := resident.Keys[(keyPos*resident.KVHeads+kvHead)*hd : (keyPos*resident.KVHeads+kvHead+1)*hd]
			var dot float64
			for i := 0; i < hd; i++ {
				dot += float64(qRow[i]) * float64(kRow[i])
			}
			scores[keyPos-start] = float32(dot * scale)
		}
		hostmath.SoftmaxInPlace(scores)
		out := context[head*hd : (head+1)*hd]
		for i := range out {
			out[i] = 0
		}
		for keyPos, prob := range scores {
			vRow := resident.Values[((start+keyPos)*resident.KVHeads+kvHead)*hd : ((start+keyPos)*resident.KVHeads+kvHead+1)*hd]
			for i := 0; i < hd; i++ {
				out[i] += prob * vRow[i]
			}
		}
		bf16RoundSlice(out)
	}

	projected := make([]float32, d)
	linearRounded(projected, context, oW, 1)
	residual := make([]float32, d)
	for c := 0; c < d; c++ {
		residual[c] = dtype.RoundBF16(tokenRow[c] + projected[c])
	}
	postNorm := make([]float32, d)
	rmsNormRounded(postNorm, residual, postW, 1, d, cfg.RMSNormEps)
	gate := make([]float32, cfg.IntermediateSize)
	up := make([]float32, cfg.IntermediateSize)
	linearRounded(gate, postNorm, gateW, 1)
	linearRounded(up, postNorm, upW, 1)
	for i := range gate {
		gate[i] = dtype.RoundBF16(float32(silu64(float64(gate[i]))))
		gate[i] = dtype.RoundBF16(gate[i] * up[i])
	}
	down := make([]float32, d)
	linearRounded(down, gate, downW, 1)
	out := make([]float32, d)
	for c := 0; c < d; c++ {
		out[c] = dtype.RoundBF16(residual[c] + down[c])
	}
	return out, nil
}

// DecodeLayerStack: one decode token through every layer (host reference).
func DecodeLayerStack(embedding []float32, decodeMask []int, segments [][2]int, resident []*ResidentKV, tokenPos int, cfg Config, weights []LayerWeights) ([]float32, error) {
	if len(resident) != cfg.NumHiddenLayers || len(weights) != cfg.NumHiddenLayers {
		return nil, fmt.Errorf("routed lm decode stack: layers resident=%d weights=%d want %d", len(resident), len(weights), cfg.NumHiddenLayers)
	}
	if tokenPos != resident[0].Tokens {
		return nil, fmt.Errorf("routed lm decode stack: token pos=%d resident=%d", tokenPos, resident[0].Tokens)
	}
	row := embedding
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		next, err := DecodeRow(row, decodeMask[:resident[layer].Tokens+1], segments, resident[layer], cfg, weights[layer])
		if err != nil {
			return nil, fmt.Errorf("routed lm decode layer %d: %w", layer, err)
		}
		row = next
	}
	return row, nil
}
