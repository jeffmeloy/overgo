package rxbrain

import (
	"fmt"
	"math"
)

// Text-only host-reference forward for the base branch: pre-RMSNorm grouped
// -query attention with per-head-dim RMS query/key layernorms and dynamic
// NTK-alpha RoPE, SwiGLU MLP, tied-embedding logits. Deterministic FP32 on
// host -- the parity anchor every later lane must match.

const rmsNormEps = 1e-5

// RopeInvFreq derives the inverse frequencies with the dynamic NTK-alpha
// scaling the checkpoint declares: base = theta * alpha^(d/(d-2)).
func RopeInvFreq(config Config, alpha float64) []float64 {
	base := config.RopeTheta
	if alpha > 0 {
		base = config.RopeTheta * math.Pow(alpha, float64(config.HeadDim)/float64(config.HeadDim-2))
	}
	half := config.HeadDim / 2
	inv := make([]float64, half)
	for i := range inv {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(config.HeadDim))
	}
	return inv
}

func rmsNorm(destination, source, weight []float32, width int) {
	rows := len(source) / width
	for row := 0; row < rows; row++ {
		segment := source[row*width : (row+1)*width]
		sum := float64(0)
		for _, value := range segment {
			sum += float64(value) * float64(value)
		}
		scale := float32(1.0 / math.Sqrt(sum/float64(width)+rmsNormEps))
		out := destination[row*width : (row+1)*width]
		for index := range segment {
			out[index] = segment[index] * scale * weight[index]
		}
	}
}

func matmulTransposed(destination, input, weight []float32, rows, inputDim, outputDim int) {
	for row := 0; row < rows; row++ {
		in := input[row*inputDim : (row+1)*inputDim]
		out := destination[row*outputDim : (row+1)*outputDim]
		for o := 0; o < outputDim; o++ {
			w := weight[o*inputDim : (o+1)*inputDim]
			sum := float64(0)
			for i := range in {
				sum += float64(in[i]) * float64(w[i])
			}
			out[o] = float32(sum)
		}
	}
}

// ForwardText runs the full base-branch stack over token ids and returns the
// final hidden states after the terminal norm, one row per token.
func ForwardText(config Config, weights TextWeights, tokens []int, ropeAlpha float64) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("rxbrain: forward requires tokens")
	}
	hidden, heads, kvHeads, headDim := config.HiddenSize, config.NumAttentionHeads, config.NumKeyValueHeads, config.HeadDim
	kvDim := kvHeads * headDim
	seq := len(tokens)
	x := make([]float32, seq*hidden)
	for position, token := range tokens {
		if token < 0 || token >= config.VocabSize {
			return nil, fmt.Errorf("rxbrain: token %d outside vocabulary", token)
		}
		copy(x[position*hidden:(position+1)*hidden], weights.Embed[token*hidden:(token+1)*hidden])
	}
	inv := RopeInvFreq(config, ropeAlpha)
	normed := make([]float32, seq*hidden)
	q := make([]float32, seq*hidden)
	k := make([]float32, seq*kvDim)
	v := make([]float32, seq*kvDim)
	attended := make([]float32, seq*hidden)
	projected := make([]float32, seq*hidden)
	gate := make([]float32, seq*config.IntermediateSize)
	up := make([]float32, seq*config.IntermediateSize)
	scores := make([]float64, seq)
	groups := heads / kvHeads
	scale := 1.0 / math.Sqrt(float64(headDim))

	for _, layer := range weights.Layers {
		rmsNorm(normed, x, layer.InputLN, hidden)
		matmulTransposed(q, normed, layer.QProj, seq, hidden, hidden)
		matmulTransposed(k, normed, layer.KProj, seq, hidden, kvDim)
		matmulTransposed(v, normed, layer.VProj, seq, hidden, kvDim)
		// Per-head RMS layernorm over head_dim, then RoPE.
		for position := 0; position < seq; position++ {
			for head := 0; head < heads; head++ {
				rmsNorm(q[position*hidden+head*headDim:position*hidden+(head+1)*headDim],
					q[position*hidden+head*headDim:position*hidden+(head+1)*headDim], layer.QueryLN, headDim)
			}
			for head := 0; head < kvHeads; head++ {
				rmsNorm(k[position*kvDim+head*headDim:position*kvDim+(head+1)*headDim],
					k[position*kvDim+head*headDim:position*kvDim+(head+1)*headDim], layer.KeyLN, headDim)
			}
			applyRope(q[position*hidden:(position+1)*hidden], heads, headDim, position, inv)
			applyRope(k[position*kvDim:(position+1)*kvDim], kvHeads, headDim, position, inv)
		}
		// Causal grouped-query attention.
		for position := 0; position < seq; position++ {
			for head := 0; head < heads; head++ {
				kvHead := head / groups
				query := q[position*hidden+head*headDim : position*hidden+(head+1)*headDim]
				maximum := math.Inf(-1)
				for past := 0; past <= position; past++ {
					key := k[past*kvDim+kvHead*headDim : past*kvDim+(kvHead+1)*headDim]
					dot := float64(0)
					for i := range query {
						dot += float64(query[i]) * float64(key[i])
					}
					scores[past] = dot * scale
					if scores[past] > maximum {
						maximum = scores[past]
					}
				}
				total := float64(0)
				for past := 0; past <= position; past++ {
					scores[past] = math.Exp(scores[past] - maximum)
					total += scores[past]
				}
				out := attended[position*hidden+head*headDim : position*hidden+(head+1)*headDim]
				for i := range out {
					out[i] = 0
				}
				for past := 0; past <= position; past++ {
					weight := float32(scores[past] / total)
					value := v[past*kvDim+kvHead*headDim : past*kvDim+(kvHead+1)*headDim]
					for i := range out {
						out[i] += weight * value[i]
					}
				}
			}
		}
		matmulTransposed(projected, attended, layer.OProj, seq, hidden, hidden)
		for i := range x {
			x[i] += projected[i]
		}
		rmsNorm(normed, x, layer.PostLN, hidden)
		matmulTransposed(gate, normed, layer.GateProj, seq, hidden, config.IntermediateSize)
		matmulTransposed(up, normed, layer.UpProj, seq, hidden, config.IntermediateSize)
		for i := range gate {
			g := float64(gate[i])
			gate[i] = float32(g / (1 + math.Exp(-g)) * float64(up[i]))
		}
		matmulTransposed(projected, gate, layer.DownProj, seq, config.IntermediateSize, hidden)
		for i := range x {
			x[i] += projected[i]
		}
	}
	rmsNorm(x, x, weights.FinalNorm, hidden)
	return x, nil
}

func applyRope(row []float32, heads, headDim, position int, inv []float64) {
	half := headDim / 2
	for head := 0; head < heads; head++ {
		segment := row[head*headDim : (head+1)*headDim]
		for i := 0; i < half; i++ {
			angle := float64(position) * inv[i]
			sin, cos := math.Sincos(angle)
			a, b := float64(segment[i]), float64(segment[i+half])
			segment[i] = float32(a*cos - b*sin)
			segment[i+half] = float32(a*sin + b*cos)
		}
	}
}

// GreedyNextToken scores the last hidden row against the tied embedding and
// returns the argmax token -- the streamed terminal the probe consumes.
func GreedyNextToken(config Config, weights TextWeights, finalHidden []float32) (int, float32) {
	hidden := config.HiddenSize
	last := finalHidden[len(finalHidden)-hidden:]
	best, bestScore := 0, float32(math.Inf(-1))
	for token := 0; token < config.VocabSize; token++ {
		row := weights.Embed[token*hidden : (token+1)*hidden]
		sum := float64(0)
		for i := range last {
			sum += float64(last[i]) * float64(row[i])
		}
		if score := float32(sum); score > bestScore {
			best, bestScore = token, score
		}
	}
	return best, bestScore
}
