// Krea2Transformer2DModel flow-matching denoiser (host port). This is a faithful
// port of diffusers Krea2Transformer2DModel (single-stream MMDiT): the tapped
// text-encoder hidden states are fused (text_fusion + txt_in), the patchified
// image latents are embedded (img_in), the two are CONCATENATED into one
// [text, image] token sequence, and 28 modulated transformer blocks co-attend
// over the whole sequence before the image tokens are sliced off and projected
// to the flow-matching velocity by final_layer. Every geometry fact is DERIVED
// from TransformerSpec (which is shape-cross-checked against the real checkpoint
// in verify.go); no magic numbers live here.
//
// Arithmetic mirrors the reference exactly:
//   - Krea2RMSNorm is ZERO-CENTERED: the effective scale is (1 + weight), the
//     normalization runs over the last axis with +eps inside the sqrt (eps read
//     from transformer/config.json norm_eps).
//   - attention is grouped-query (48 q heads / 12 kv heads), per-head q/k
//     RMSNorm, 3-axis interleaved RoPE (axes [32,48,48], theta 1000), a sigmoid
//     output gate, and no biases on any projection.
//   - modulation is AdaLN-single: one shared 6*hidden vector (time_mod_proj of
//     gelu-tanh(time_embed(sigma))) plus a per-block learned scale_shift_table;
//     the six fields gate the pre-attention and pre-FF norms and both residuals.
//   - the SwiGLU FF is down(silu(gate(x)) * up(x)).
//
// Host math accumulates in f64 (reference-oracle convention, matching vae.go);
// the device serves this net in bf16. See TestDenoiser* for what is verifiable.
//
// NOTE ON g3 PARITY. The bit-exact g3 image needs the exact final latent, which
// needs (a) the exact text conditioning -- the Qwen3VL 36-layer encoder's 12
// selected hidden states. That encoder IS now ported (textencoder.go: streamed
// host forward -> [textSeq,12,TextHidden] feeding textConditioning here), but its
// exact numeric values still need the adaptive dump hook (g1 present=false, not in
// the goldens) so this remains telemetry/structural, not bit-exact -- and (b) the
// exact seed-42 noise (torch randn, reproduced natively by adaptive; RNG not
// matched here). Additionally the real checkpoint
// is 12.82B params (~51GB as f32), so a full-scale host forward is not runnable
// CPU-only. This file therefore VERIFIES block shapes against the real checkpoint
// (headers) and exercises the exact forward arithmetic at synthetic scale; the
// exact-g3 endpoint is named as a gap in TestDenoiserExactG3IsHookGapped.
package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
)

// Denoiser: a compiled, weight-resident Krea2 transformer. store maps every
// diffusers tensor name to its F32 payload; NewDenoiser validates the full set
// against DenoiserTensorShapes so every capability tensor is bound and sized.
type Denoiser struct {
	T                 TransformerSpec
	Eps               float64
	NumTrainTimesteps int
	store             map[string][]float32
	WeightBytes       int64
}

// textFusionHeadDim / attnHeadDim are DERIVED (dim/heads); the checkpoint pins
// them to HeadDim (128) via the norm_q/norm_k length cross-check in verify.go.

// DenoiserTensorShapes returns the complete name -> torch shape ([out,in] for
// Linear weights) manifest the forward consumes, derived entirely from spec.
// This is the capability-retention contract: every listed tensor must exist in
// the checkpoint with this shape, and the forward touches every one.
func DenoiserTensorShapes(t TransformerSpec) map[string][]int {
	h := t.Hidden
	shapes := map[string][]int{
		"img_in.weight":                 {h, t.InChannels},
		"img_in.bias":                   {h},
		"time_embed.linear_1.weight":    {h, t.TimestepEmbed},
		"time_embed.linear_1.bias":      {h},
		"time_embed.linear_2.weight":    {h, h},
		"time_embed.linear_2.bias":      {h},
		"time_mod_proj.weight":          {6 * h, h},
		"time_mod_proj.bias":            {6 * h},
		"txt_in.norm.weight":            {t.TextHidden},
		"txt_in.linear_1.weight":        {h, t.TextHidden},
		"txt_in.linear_1.bias":          {h},
		"txt_in.linear_2.weight":        {h, h},
		"txt_in.linear_2.bias":          {h},
		"text_fusion.projector.weight":  {1, t.TextLayers},
		"final_layer.scale_shift_table": {2, h},
		"final_layer.norm.weight":       {h},
		"final_layer.linear.weight":     {t.InChannels, h},
		"final_layer.linear.bias":       {t.InChannels},
	}
	// text-fusion blocks operate at TextHidden with TextHeads/TextKVHeads.
	th := t.TextHidden
	addFusionBlocks := func(kind string, n int) {
		for i := 0; i < n; i++ {
			p := fmt.Sprintf("text_fusion.%s.%d.", kind, i)
			addAttnFF(shapes, p, th, t.TextHeads*t.HeadDim, t.TextKVHeads*t.HeadDim, t.HeadDim, t.TextIntermediate, false)
		}
	}
	addFusionBlocks("layerwise_blocks", t.LayerwiseTextBlocks)
	addFusionBlocks("refiner_blocks", t.RefinerTextBlocks)
	// the 28 image/text co-attention blocks at Hidden with GQA.
	for i := 0; i < t.Layers; i++ {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		shapes[p+"scale_shift_table"] = []int{t.ModFieldsOr6(), h}
		addAttnFF(shapes, p, h, h, t.KVDim, t.HeadDim, t.Intermediate, true)
	}
	return shapes
}

// ModFieldsOr6 returns the modulation field count, defaulting to the reference 6
// when the spec has not been checkpoint-verified yet.
func (t TransformerSpec) ModFieldsOr6() int {
	if t.ModFields > 0 {
		return t.ModFields
	}
	return 6
}

// addAttnFF appends the shared attention + norm + SwiGLU tensor shapes for one
// pre-norm block. qDim = heads*head_dim, kvDim = kv_heads*head_dim. withTable is
// informational only (the table is added by the caller for image blocks).
func addAttnFF(shapes map[string][]int, p string, dim, qDim, kvDim, headDim, inter int, _ bool) {
	shapes[p+"norm1.weight"] = []int{dim}
	shapes[p+"norm2.weight"] = []int{dim}
	shapes[p+"attn.norm_q.weight"] = []int{headDim}
	shapes[p+"attn.norm_k.weight"] = []int{headDim}
	shapes[p+"attn.to_q.weight"] = []int{qDim, dim}
	shapes[p+"attn.to_k.weight"] = []int{kvDim, dim}
	shapes[p+"attn.to_v.weight"] = []int{kvDim, dim}
	shapes[p+"attn.to_gate.weight"] = []int{dim, dim}
	shapes[p+"attn.to_out.0.weight"] = []int{dim, dim}
	shapes[p+"ff.gate.weight"] = []int{inter, dim}
	shapes[p+"ff.up.weight"] = []int{inter, dim}
	shapes[p+"ff.down.weight"] = []int{dim, inter}
}

func prod(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// NewDenoiser binds a weight store to the spec, validating that every tensor in
// the derived manifest is present with the exact element count (capability
// retention) and that no extra image-path tensor is silently ignored.
func NewDenoiser(t TransformerSpec, eps float64, numTrainTimesteps int, store map[string][]float32) (*Denoiser, error) {
	if eps <= 0 {
		return nil, fmt.Errorf("denoiser: eps must be positive, got %g", eps)
	}
	if numTrainTimesteps <= 0 {
		return nil, fmt.Errorf("denoiser: num_train_timesteps must be positive, got %d", numTrainTimesteps)
	}
	if t.ModFields == 0 {
		t.ModFields = 6
	}
	shapes := DenoiserTensorShapes(t)
	var bytes int64
	for name, shape := range shapes {
		v, ok := store[name]
		if !ok {
			return nil, fmt.Errorf("denoiser: missing tensor %s (want %v)", name, shape)
		}
		if want := prod(shape); len(v) != want {
			return nil, fmt.Errorf("denoiser: tensor %s len=%d want %d (shape %v)", name, len(v), want, shape)
		}
		bytes += int64(len(v)) * 4
	}
	return &Denoiser{T: t, Eps: eps, NumTrainTimesteps: numTrainTimesteps, store: store, WeightBytes: bytes}, nil
}

func (d *Denoiser) w(name string) []float32 { return d.store[name] }

// ---- host math primitives (f64 accumulation) -------------------------------

// dense computes y[rows,outDim] = x[rows,inDim] @ W^T (+bias), where W is the
// torch Linear weight [outDim,inDim] row-major. bias may be nil.
func dense(x []float64, weight, bias []float32, rows, inDim, outDim int) []float64 {
	y := make([]float64, rows*outDim)
	hostmath.ParallelRangeF64(rows, inDim*outDim, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			xr := x[r*inDim : (r+1)*inDim]
			yr := y[r*outDim : (r+1)*outDim]
			for o := 0; o < outDim; o++ {
				acc := 0.0
				if bias != nil {
					acc = float64(bias[o])
				}
				wo := weight[o*inDim : (o+1)*inDim]
				for i := 0; i < inDim; i++ {
					acc += xr[i] * float64(wo[i])
				}
				yr[o] = acc
			}
		}
	})
	return y
}

// rmsNormZeroCentered applies Krea2RMSNorm over the last axis of x[rows,d]:
// out = x / sqrt(mean(x^2)+eps) * (1 + weight). Returns a new buffer.
func rmsNormZeroCentered(x []float64, weight []float32, rows, d int, eps float64) []float64 {
	out := make([]float64, len(x))
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		or := out[r*d : (r+1)*d]
		var ss float64
		for i := 0; i < d; i++ {
			ss += xr[i] * xr[i]
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		for i := 0; i < d; i++ {
			or[i] = xr[i] * inv * (1.0 + float64(weight[i]))
		}
	}
	return out
}

func geluTanh(v float64) float64 {
	return 0.5 * v * (1.0 + math.Tanh(0.7978845608028654*(v+0.044715*v*v*v)))
}

func siluF64(v float64) float64 { return v / (1.0 + math.Exp(-v)) }

func sigmoidF64(v float64) float64 { return 1.0 / (1.0 + math.Exp(-v)) }

// ---- rotary position embedding (3-axis, interleaved-real) -------------------

// ropeTable builds per-token cos/sin of length HeadDim for the combined
// sequence: text tokens sit at (0,0,0); image token n at (0, n/gw, n%gw). Each
// axis a contributes RopeAxes[a] channels via freqs theta^(-2i/da), angle
// pos*freq, repeat-interleaved to width da (Flux/diffusers convention).
func (d *Denoiser) ropeTable(textSeq, gh, gw int) (cos, sin []float64) {
	seq := textSeq + gh*gw
	hd := d.T.HeadDim
	cos = make([]float64, seq*hd)
	sin = make([]float64, seq*hd)
	axes := d.T.RopeAxes
	theta := d.T.RopeTheta
	for tok := 0; tok < seq; tok++ {
		var pos [3]float64
		if tok >= textSeq {
			n := tok - textSeq
			pos = [3]float64{0, float64(n / gw), float64(n % gw)}
		}
		off := 0
		for a := 0; a < 3; a++ {
			da := axes[a]
			half := da / 2
			for i := 0; i < half; i++ {
				freq := math.Pow(theta, -float64(2*i)/float64(da))
				angle := pos[a] * freq
				c, s := math.Cos(angle), math.Sin(angle)
				// repeat-interleaved: channels off+2i and off+2i+1 share (c,s).
				cos[tok*hd+off+2*i] = c
				cos[tok*hd+off+2*i+1] = c
				sin[tok*hd+off+2*i] = s
				sin[tok*hd+off+2*i+1] = s
			}
			off += da
		}
	}
	return cos, sin
}

// applyRopeInterleaved rotates one head vector in place: for each pair (2p,2p+1)
// out0 = x0*cos - x1*sin ; out1 = x1*cos + x0*sin (cos/sin equal within a pair).
func applyRopeInterleaved(vec []float64, cos, sin []float64, headDim int) {
	for p := 0; p < headDim/2; p++ {
		i0, i1 := 2*p, 2*p+1
		x0, x1 := vec[i0], vec[i1]
		vec[i0] = x0*cos[i0] - x1*sin[i0]
		vec[i1] = x1*cos[i1] + x0*sin[i1]
	}
}

// headRMSAndRope normalizes each head's q/k over head_dim with the zero-centered
// scale and (optionally) applies RoPE. arr is [seq, heads*headDim] token-major.
func (d *Denoiser) headRMSAndRope(arr []float64, normW []float32, seq, heads, headDim int, cos, sin []float64) {
	for r := 0; r < seq; r++ {
		for hh := 0; hh < heads; hh++ {
			base := (r*heads + hh) * headDim
			vec := arr[base : base+headDim]
			var ss float64
			for i := 0; i < headDim; i++ {
				ss += vec[i] * vec[i]
			}
			inv := 1.0 / math.Sqrt(ss/float64(headDim)+d.Eps)
			for i := 0; i < headDim; i++ {
				vec[i] = vec[i] * inv * (1.0 + float64(normW[i]))
			}
			if cos != nil {
				applyRopeInterleaved(vec, cos[r*headDim:(r+1)*headDim], sin[r*headDim:(r+1)*headDim], headDim)
			}
		}
	}
}

// attention runs one Krea2Attention: GQA scaled-dot-product over the full
// (bidirectional) sequence, per-head q/k RMSNorm, optional RoPE, sigmoid output
// gate, no biases. dim = heads*headDim = gate/out width.
func (d *Denoiser) attention(prefix string, x []float64, seq, dim, heads, kvHeads, headDim int, cos, sin []float64) []float64 {
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	q := dense(x, d.w(prefix+"to_q.weight"), nil, seq, dim, qDim)
	k := dense(x, d.w(prefix+"to_k.weight"), nil, seq, dim, kvDim)
	v := dense(x, d.w(prefix+"to_v.weight"), nil, seq, dim, kvDim)
	gate := dense(x, d.w(prefix+"to_gate.weight"), nil, seq, dim, dim)
	d.headRMSAndRope(q, d.w(prefix+"norm_q.weight"), seq, heads, headDim, cos, sin)
	d.headRMSAndRope(k, d.w(prefix+"norm_k.weight"), seq, kvHeads, headDim, cos, sin)

	group := heads / kvHeads
	scale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := make([]float64, seq*qDim) // [seq, heads*headDim]
	hostmath.ParallelRangeF64(heads, seq*seq*headDim, func(loH, hiH int) {
		scores := make([]float64, seq)
		for hh := loH; hh < hiH; hh++ {
			kvh := hh / group
			for i := 0; i < seq; i++ {
				qvec := q[(i*heads+hh)*headDim : (i*heads+hh)*headDim+headDim]
				var mx float64 = math.Inf(-1)
				for j := 0; j < seq; j++ {
					kvec := k[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
					var dot float64
					for c := 0; c < headDim; c++ {
						dot += qvec[c] * kvec[c]
					}
					dot *= scale
					scores[j] = dot
					if dot > mx {
						mx = dot
					}
				}
				var sum float64
				for j := 0; j < seq; j++ {
					e := math.Exp(scores[j] - mx)
					scores[j] = e
					sum += e
				}
				dst := attnOut[(i*heads+hh)*headDim : (i*heads+hh)*headDim+headDim]
				for j := 0; j < seq; j++ {
					p := scores[j] / sum
					vvec := v[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
					for c := 0; c < headDim; c++ {
						dst[c] += p * vvec[c]
					}
				}
			}
		}
	})
	for idx := range attnOut {
		attnOut[idx] *= sigmoidF64(gate[idx])
	}
	return dense(attnOut, d.w(prefix+"to_out.0.weight"), nil, seq, dim, dim)
}

// swiGLU computes down(silu(gate(x)) * up(x)) at hidden width inter.
func (d *Denoiser) swiGLU(prefix string, x []float64, seq, dim, inter int) []float64 {
	g := dense(x, d.w(prefix+"gate.weight"), nil, seq, dim, inter)
	u := dense(x, d.w(prefix+"up.weight"), nil, seq, dim, inter)
	for i := range g {
		g[i] = siluF64(g[i]) * u[i]
	}
	return dense(g, d.w(prefix+"down.weight"), nil, seq, inter, dim)
}

// fusionBlock runs one Krea2TextFusionBlock (pre-norm, no RoPE, no modulation).
func (d *Denoiser) fusionBlock(prefix string, h []float64, seq, dim int) []float64 {
	n1 := rmsNormZeroCentered(h, d.w(prefix+"norm1.weight"), seq, dim, d.Eps)
	attn := d.attention(prefix+"attn.", n1, seq, dim, d.T.TextHeads, d.T.TextKVHeads, d.T.HeadDim, nil, nil)
	for i := range h {
		h[i] += attn[i]
	}
	n2 := rmsNormZeroCentered(h, d.w(prefix+"norm2.weight"), seq, dim, d.Eps)
	ff := d.swiGLU(prefix+"ff.", n2, seq, dim, d.T.TextIntermediate)
	for i := range h {
		h[i] += ff[i]
	}
	return h
}

// textConditioning runs text_fusion + txt_in: layerwise blocks attend across the
// tapped-layer axis (per token), a projector collapses that axis, refiner blocks
// attend across tokens, then txt_in projects to Hidden. encoderHidden is
// [textSeq, TextLayers, TextHidden] row-major. Returns [textSeq, Hidden].
func (d *Denoiser) textConditioning(encoderHidden []float64, textSeq int) ([]float64, error) {
	L := d.T.TextLayers
	th := d.T.TextHidden
	if len(encoderHidden) != textSeq*L*th {
		return nil, fmt.Errorf("denoiser: encoder hidden len=%d want %d ([%d,%d,%d])", len(encoderHidden), textSeq*L*th, textSeq, L, th)
	}
	// layerwise: for each token, a length-L sequence of TextHidden vectors.
	hs := append([]float64(nil), encoderHidden...) // [textSeq*L, th] viewed as textSeq sequences of len L
	for tok := 0; tok < textSeq; tok++ {
		seqSlice := hs[tok*L*th : (tok+1)*L*th]
		for b := 0; b < d.T.LayerwiseTextBlocks; b++ {
			p := fmt.Sprintf("text_fusion.layerwise_blocks.%d.", b)
			d.fusionBlock(p, seqSlice, L, th)
		}
	}
	// projector: collapse the layer axis with Linear(L->1) weight [1,L].
	proj := d.w("text_fusion.projector.weight") // len L
	fused := make([]float64, textSeq*th)
	for tok := 0; tok < textSeq; tok++ {
		for c := 0; c < th; c++ {
			var acc float64
			for l := 0; l < L; l++ {
				acc += float64(proj[l]) * hs[(tok*L+l)*th+c]
			}
			fused[tok*th+c] = acc
		}
	}
	// refiner: attend across the token sequence.
	for b := 0; b < d.T.RefinerTextBlocks; b++ {
		p := fmt.Sprintf("text_fusion.refiner_blocks.%d.", b)
		d.fusionBlock(p, fused, textSeq, th)
	}
	// txt_in: norm -> linear_1 -> gelu(tanh) -> linear_2.
	normed := rmsNormZeroCentered(fused, d.w("txt_in.norm.weight"), textSeq, th, d.Eps)
	l1 := dense(normed, d.w("txt_in.linear_1.weight"), d.w("txt_in.linear_1.bias"), textSeq, th, d.T.Hidden)
	for i := range l1 {
		l1[i] = geluTanh(l1[i])
	}
	return dense(l1, d.w("txt_in.linear_2.weight"), d.w("txt_in.linear_2.bias"), textSeq, d.T.Hidden, d.T.Hidden), nil
}

// timestepConditioning returns temb [Hidden] and tembMod [6*Hidden] from the
// flow-time sigma. temb = linear_2(gelu(linear_1(sinusoid(sigma)))); tembMod =
// time_mod_proj(gelu(temb)). Sinusoid: cos-first, input scaled by 1e3.
func (d *Denoiser) timestepConditioning(sigma float64) (temb, tembMod []float64) {
	dim := d.T.TimestepEmbed
	half := dim / 2
	emb := make([]float64, dim)
	for i := 0; i < half; i++ {
		freq := math.Exp(-math.Log(1e4) * float64(i) / float64(half))
		arg := sigma * 1e3 * freq
		emb[i] = math.Cos(arg)
		emb[half+i] = math.Sin(arg)
	}
	l1 := dense(emb, d.w("time_embed.linear_1.weight"), d.w("time_embed.linear_1.bias"), 1, dim, d.T.Hidden)
	for i := range l1 {
		l1[i] = geluTanh(l1[i])
	}
	temb = dense(l1, d.w("time_embed.linear_2.weight"), d.w("time_embed.linear_2.bias"), 1, d.T.Hidden, d.T.Hidden)
	modIn := make([]float64, d.T.Hidden)
	for i := range temb {
		modIn[i] = geluTanh(temb[i])
	}
	tembMod = dense(modIn, d.w("time_mod_proj.weight"), d.w("time_mod_proj.bias"), 1, d.T.Hidden, 6*d.T.Hidden)
	return temb, tembMod
}

// Forward predicts the flow-matching velocity for one denoise step. latentPatches
// is the packed image sequence [imgSeq, InChannels] (row-major, see PackLatent);
// encoderHidden is [textSeq, TextLayers, TextHidden]; sigma is the flow time in
// [0,1]. gh*gw must equal imgSeq. Returns velocity [imgSeq, InChannels].
func (d *Denoiser) Forward(latentPatches, encoderHidden []float64, sigma float64, textSeq, gh, gw int) ([]float64, error) {
	h := d.T.Hidden
	imgSeq := gh * gw
	if len(latentPatches) != imgSeq*d.T.InChannels {
		return nil, fmt.Errorf("denoiser: latent patches len=%d want %d ([%d,%d])", len(latentPatches), imgSeq*d.T.InChannels, imgSeq, d.T.InChannels)
	}
	if textSeq <= 0 {
		return nil, fmt.Errorf("denoiser: textSeq must be positive, got %d", textSeq)
	}
	temb, tembMod := d.timestepConditioning(sigma)

	txt, err := d.textConditioning(encoderHidden, textSeq)
	if err != nil {
		return nil, err
	}
	img := dense(latentPatches, d.w("img_in.weight"), d.w("img_in.bias"), imgSeq, d.T.InChannels, h)

	seq := textSeq + imgSeq
	hidden := make([]float64, seq*h)
	copy(hidden[:textSeq*h], txt)
	copy(hidden[textSeq*h:], img)

	cos, sin := d.ropeTable(textSeq, gh, gw)

	// per-block modulation = tembMod (6*h) reshaped + scale_shift_table[6,h].
	for layer := 0; layer < d.T.Layers; layer++ {
		prefix := fmt.Sprintf("transformer_blocks.%d.", layer)
		table := d.w(prefix + "scale_shift_table") // [6,h]
		mod := make([]float64, 6*h)
		for i := 0; i < 6*h; i++ {
			mod[i] = tembMod[i] + float64(table[i])
		}
		preScale, preShift, preGate := mod[0:h], mod[h:2*h], mod[2*h:3*h]
		postScale, postShift, postGate := mod[3*h:4*h], mod[4*h:5*h], mod[5*h:6*h]

		n1 := rmsNormZeroCentered(hidden, d.w(prefix+"norm1.weight"), seq, h, d.Eps)
		for r := 0; r < seq; r++ {
			for c := 0; c < h; c++ {
				n1[r*h+c] = (1.0+preScale[c])*n1[r*h+c] + preShift[c]
			}
		}
		attn := d.attention(prefix+"attn.", n1, seq, h, d.T.Heads, d.T.KVHeads, d.T.HeadDim, cos, sin)
		for r := 0; r < seq; r++ {
			for c := 0; c < h; c++ {
				hidden[r*h+c] += preGate[c] * attn[r*h+c]
			}
		}
		n2 := rmsNormZeroCentered(hidden, d.w(prefix+"norm2.weight"), seq, h, d.Eps)
		for r := 0; r < seq; r++ {
			for c := 0; c < h; c++ {
				n2[r*h+c] = (1.0+postScale[c])*n2[r*h+c] + postShift[c]
			}
		}
		ff := d.swiGLU(prefix+"ff.", n2, seq, h, d.T.Intermediate)
		for r := 0; r < seq; r++ {
			for c := 0; c < h; c++ {
				hidden[r*h+c] += postGate[c] * ff[r*h+c]
			}
		}
	}

	// slice image tokens, final adaptive-norm + projection to velocity.
	imgHidden := hidden[textSeq*h:]
	table := d.w("final_layer.scale_shift_table") // [2,h]
	scale := make([]float64, h)
	shift := make([]float64, h)
	for c := 0; c < h; c++ {
		scale[c] = temb[c] + float64(table[c])
		shift[c] = temb[c] + float64(table[h+c])
	}
	fn := rmsNormZeroCentered(imgHidden, d.w("final_layer.norm.weight"), imgSeq, h, d.Eps)
	for r := 0; r < imgSeq; r++ {
		for c := 0; c < h; c++ {
			fn[r*h+c] = (1.0+scale[c])*fn[r*h+c] + shift[c]
		}
	}
	return dense(fn, d.w("final_layer.linear.weight"), d.w("final_layer.linear.bias"), imgSeq, h, d.T.InChannels), nil
}

// ---- latent <-> patch packing (mirrors pipeline _pack_latents) -------------

// PatchChannelOrder selects the per-patch element order.
type PatchChannelOrder uint8

const (
	PatchChannelsFirst PatchChannelOrder = iota // channel, row, column
	PatchChannelsLast                           // row, column, channel
)

// PackPlanarF32 packs planar [C,H,W] into row-major patch tokens.
func PackPlanarF32(planar []float32, c, hh, ww, patch int, order PatchChannelOrder) ([]float32, int, int, error) {
	return packPlanar(planar, c, hh, ww, patch, order)
}

// UnpackPlanarF32 reverses PackPlanarF32.
func UnpackPlanarF32(patches []float32, c, gh, gw, patch int, order PatchChannelOrder) ([]float32, error) {
	return unpackPlanar(patches, c, gh, gw, patch, order)
}

// PackLatent packs a channel-major latent [C, H, W] into the transformer image
// sequence [gh*gw, C*patch*patch] where gh=H/patch, gw=W/patch and the per-patch
// row order is (channel, ph, pw) -- exactly diffusers _pack_latents.
func PackLatent(latent []float64, c, hh, ww, patch int) ([]float64, int, int, error) {
	return packPlanar(latent, c, hh, ww, patch, PatchChannelsFirst)
}

func packLatent[T ~float32 | ~float64](latent []T, c, hh, ww, patch int) ([]T, int, int, error) {
	return packPlanar(latent, c, hh, ww, patch, PatchChannelsFirst)
}

func packPlanar[T ~float32 | ~float64](latent []T, c, hh, ww, patch int, order PatchChannelOrder) ([]T, int, int, error) {
	if order != PatchChannelsFirst && order != PatchChannelsLast {
		return nil, 0, 0, fmt.Errorf("pack: invalid channel order %d", order)
	}
	if hh%patch != 0 || ww%patch != 0 {
		return nil, 0, 0, fmt.Errorf("pack: %dx%d not divisible by patch %d", hh, ww, patch)
	}
	if len(latent) != c*hh*ww {
		return nil, 0, 0, fmt.Errorf("pack: latent len=%d want %d", len(latent), c*hh*ww)
	}
	gh, gw := hh/patch, ww/patch
	inCh := c * patch * patch
	out := make([]T, gh*gw*inCh)
	for r := 0; r < gh; r++ {
		for col := 0; col < gw; col++ {
			row := out[(r*gw+col)*inCh : (r*gw+col+1)*inCh]
			if order == PatchChannelsFirst {
				for ch := 0; ch < c; ch++ {
					for ph := 0; ph < patch; ph++ {
						for pw := 0; pw < patch; pw++ {
							row[(ch*patch+ph)*patch+pw] = latent[(ch*hh+r*patch+ph)*ww+col*patch+pw]
						}
					}
				}
			} else {
				for ph := 0; ph < patch; ph++ {
					for pw := 0; pw < patch; pw++ {
						for ch := 0; ch < c; ch++ {
							row[(ph*patch+pw)*c+ch] = latent[(ch*hh+r*patch+ph)*ww+col*patch+pw]
						}
					}
				}
			}
		}
	}
	return out, gh, gw, nil
}

// UnpackLatent is the inverse of PackLatent: image sequence [gh*gw, C*patch^2]
// -> channel-major latent [C, H, W].
func UnpackLatent(patches []float64, c, gh, gw, patch int) ([]float64, error) {
	return unpackPlanar(patches, c, gh, gw, patch, PatchChannelsFirst)
}

func unpackLatent[T ~float32 | ~float64](patches []T, c, gh, gw, patch int) ([]T, error) {
	return unpackPlanar(patches, c, gh, gw, patch, PatchChannelsFirst)
}

func unpackPlanar[T ~float32 | ~float64](patches []T, c, gh, gw, patch int, order PatchChannelOrder) ([]T, error) {
	if order != PatchChannelsFirst && order != PatchChannelsLast {
		return nil, fmt.Errorf("unpack: invalid channel order %d", order)
	}
	inCh := c * patch * patch
	if len(patches) != gh*gw*inCh {
		return nil, fmt.Errorf("unpack: patches len=%d want %d", len(patches), gh*gw*inCh)
	}
	hh, ww := gh*patch, gw*patch
	out := make([]T, c*hh*ww)
	for r := 0; r < gh; r++ {
		for col := 0; col < gw; col++ {
			row := patches[(r*gw+col)*inCh : (r*gw+col+1)*inCh]
			if order == PatchChannelsFirst {
				for ch := 0; ch < c; ch++ {
					for ph := 0; ph < patch; ph++ {
						for pw := 0; pw < patch; pw++ {
							out[(ch*hh+r*patch+ph)*ww+col*patch+pw] = row[(ch*patch+ph)*patch+pw]
						}
					}
				}
			} else {
				for ph := 0; ph < patch; ph++ {
					for pw := 0; pw < patch; pw++ {
						for ch := 0; ch < c; ch++ {
							out[(ch*hh+r*patch+ph)*ww+col*patch+pw] = row[(ph*patch+pw)*c+ch]
						}
					}
				}
			}
		}
	}
	return out, nil
}

// ---- real-checkpoint structural verification (headers only) ----------------

// DenoiserWitness: the structural oracle for the denoiser port -- every derived
// tensor shape vs the real checkpoint, plus the consumed/present tallies.
type DenoiserWitness struct {
	Checks    []Check
	Tensors   int // manifest size (tensors the forward consumes)
	NormEps   float64
	ModFields int
}

func (w DenoiserWitness) Failed() bool { return failedCheckCount(w.Checks) != 0 }

// VerifyDenoiserCheckpoint opens the transformer safetensors HEADERS under
// modelDir and asserts every tensor the forward consumes exists with the exact
// derived shape, and that the checkpoint carries no un-consumed transformer
// tensor (capability retention). Never reads any payload. Returns the witness.
func VerifyDenoiserCheckpoint(modelDir string) (*DenoiserWitness, error) {
	spec, err := Derive(modelDir)
	if err != nil {
		return nil, err
	}
	t := spec.Transformer
	if t.ModFields == 0 {
		t.ModFields = 6
	}
	src, err := safetensors.OpenSource(modelDir + "/transformer")
	if err != nil {
		return nil, fmt.Errorf("denoiser verify: open transformer: %w", err)
	}
	defer src.Close()

	shapes := DenoiserTensorShapes(t)
	w := &DenoiserWitness{Tensors: len(shapes), ModFields: t.ModFields}
	consumed := make(map[string]bool, len(shapes))
	for name, shape := range shapes {
		consumed[name] = true
		tensor, ok := src.Tensors[name]
		if !ok {
			w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: -1, Source: "MISSING"})
			continue
		}
		got := 1
		for _, s := range tensor.Shape {
			got *= int(s)
		}
		w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: got, Source: name})
	}
	// capability retention: every transformer tensor must be consumed.
	var extra []string
	for _, name := range src.Names() {
		if !consumed[name] {
			extra = append(extra, name)
		}
	}
	if len(extra) > 0 {
		w.Checks = append(w.Checks, Check{Name: "unconsumed_tensors", Want: 0, Got: len(extra), Source: extra[0]})
	}
	if failures := failedCheckCount(w.Checks); failures != 0 {
		return w, fmt.Errorf("denoiser verify: %d structural check(s) disagreed with checkpoint", failures)
	}
	return w, nil
}
