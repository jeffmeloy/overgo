package speechsynth

import (
	"context"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// codecState retains one recording's bounded convolution and attention history.
// A failed decode poisons the state because its layers may have advanced unevenly.
type codecState struct {
	codec                            *CodecDecoder
	history                          [][]float32
	workspace                        [][]float32
	k, v                             [][]float32
	position                         int
	qkv, xn, attn, proj, hidden, cur []float32
	err                              error
}

func newCodecState(c *CodecDecoder) *codecState {
	s := &codecState{codec: c, history: make([][]float32, 1+len(c.convs)+tensor.PairedExtent*len(c.resnets)), k: make([][]float32, len(c.layers)), v: make([][]float32, len(c.layers)), qkv: make([]float32, tensor.TripleExtent*c.d), xn: make([]float32, c.d), attn: make([]float32, c.d), proj: make([]float32, c.d), hidden: make([]float32, c.ff)}
	s.workspace = make([][]float32, len(s.history))
	s.cur = make([]float32, c.d)
	for i := range s.k {
		// Mirrored rings expose each chronological window without shifting it.
		s.k[i] = make([]float32, tensor.PairedExtent*c.window*c.d)
		s.v[i] = make([]float32, len(s.k[i]))
	}
	return s
}

func (s *codecState) cause(ctx context.Context) error {
	if s.err == nil {
		s.err = context.Cause(ctx)
	}
	return s.err
}

// convolve preserves input overlap so float64 accumulation retains its order.
// Only kernel-sized history is replayed and the recording prefix is never retained.
func (s *codecState) convolve(index int, cv codecConv, groups int, x []float32, frames int) ([]float32, int) {
	previous := s.history[index]
	old := len(previous) / cv.cIn
	joined := x
	if old != 0 {
		count := cv.cIn * (old + frames)
		if cap(s.workspace[index]) < count {
			s.workspace[index] = make([]float32, count)
		}
		joined = s.workspace[index][:count]
		for ch := range cv.cIn {
			row := joined[ch*(old+frames) : (ch+1)*(old+frames)]
			copy(row, previous[ch*old:(ch+1)*old])
			copy(row[old:], x[ch*frames:(ch+1)*frames])
		}
	}
	keep := cv.k - cv.strd
	var out []float32
	var skip, count int
	if cv.transpose {
		keep = (keep + cv.strd - 1) / cv.strd
		skip, count = old*cv.strd, frames*cv.strd
		out = hostmath.ConvTranspose1dSuffix(joined, cv.cIn, old+frames, cv.w, cv.b, cv.cOut, cv.k, cv.strd, groups, skip)
	} else {
		skip, count = old/cv.strd, frames/cv.strd
		out = hostmath.CausalConv1dSuffix(joined, cv.cIn, old+frames, cv.w, cv.b, cv.cOut, cv.k, cv.strd, skip)
	}
	keep = min(keep, old+frames)
	if cap(previous) < cv.cIn*keep {
		previous = make([]float32, cv.cIn*keep)
	}
	previous = previous[:cv.cIn*keep]
	for ch := range cv.cIn {
		copy(previous[ch*keep:(ch+1)*keep], joined[(ch+1)*(old+frames)-keep:(ch+1)*(old+frames)])
	}
	s.history[index] = previous
	return out, count
}

func (s *codecState) transform(ctx context.Context, x []float32, frames int) error {
	c := s.codec
	d := c.d
	cur := s.cur
	for frame := range frames {
		if err := s.cause(ctx); err != nil {
			return err
		}
		for ch := range d {
			cur[ch] = x[ch*frames+frame]
		}
		t := s.position
		write := t % c.window * d
		count := min(t+1, c.window)
		start := (t + 1 - count) % c.window * d
		for li := range c.layers {
			l := &c.layers[li]
			hostmath.LayerNormInto(s.xn, cur, l.norm1W, l.norm1B, 1, d, c.normalization.TransformerLayer)
			hostmath.Linear(s.qkv, s.xn, l.inProj, 1, d, tensor.TripleExtent*d)
			q, k, v := s.qkv[:d], s.qkv[d:tensor.PairedExtent*d], s.qkv[tensor.PairedExtent*d:]
			for head := range c.h {
				hostmath.ApplyRotaryInterleaved(q[head*c.hd:(head+1)*c.hd], c.invFreq, t)
				hostmath.ApplyRotaryInterleaved(k[head*c.hd:(head+1)*c.hd], c.invFreq, t)
			}
			for _, offset := range []int{write, write + c.window*d} {
				copy(s.k[li][offset:], k)
				copy(s.v[li][offset:], v)
			}
			for i := range q {
				q[i] *= c.scoreScale
			}
			hostmath.CausalAttentionStep(s.attn, q, s.k[li][start:start+count*d], s.v[li][start:start+count*d], count, c.h, c.h, c.hd)
			hostmath.Linear(s.proj, s.attn, l.outProj, 1, d, d)
			for i := range cur {
				cur[i] += l.ls1[i] * s.proj[i]
			}
			hostmath.LayerNormInto(s.xn, cur, l.norm2W, l.norm2B, 1, d, c.normalization.TransformerLayer)
			hostmath.Linear(s.hidden, s.xn, l.lin1, 1, d, c.ff)
			hostmath.GELUErfInPlace(s.hidden)
			hostmath.Linear(s.proj, s.hidden, l.lin2, 1, c.ff, d)
			for i := range cur {
				cur[i] += l.ls2[i] * s.proj[i]
			}
		}
		for ch := range d {
			x[ch*frames+frame] = cur[ch]
		}
		s.position++
	}
	return s.cause(ctx)
}

func (s *codecState) seanet(ctx context.Context, x []float32, frames int) ([]float32, error) {
	c := s.codec
	index := 1
	apply := func(cv codecConv, input []float32) []float32 {
		out, count := s.convolve(index, cv, 1, input, frames)
		index++
		frames = count
		return out
	}
	x = apply(c.convs[0], x)
	hostmath.ELUInPlace(x)
	for i, cv := range c.convs[1 : len(c.convs)-1] {
		if err := s.cause(ctx); err != nil {
			return nil, err
		}
		x = apply(cv, x)
		v := append([]float32(nil), x...)
		for _, residual := range c.resnets[i] {
			hostmath.ELUInPlace(v)
			v = apply(residual, v)
		}
		for i := range v {
			v[i] += x[i]
		}
		x = v
		hostmath.ELUInPlace(x)
	}
	if err := s.cause(ctx); err != nil {
		return nil, err
	}
	x = apply(c.convs[len(c.convs)-1], x)
	if err := s.cause(ctx); err != nil {
		return nil, err
	}
	return x, nil
}

// decode emits every complete sample and trims the final transpose overlap.
// An empty final chunk emits nothing because each chunk already excludes that tail.
func (s *codecState) decode(ctx context.Context, latent []float32, frames int) ([]float32, error) {
	if err := s.cause(ctx); err != nil {
		return nil, err
	}
	if frames == 0 {
		return nil, nil
	}
	up, count := s.upsample(latent, frames)
	if err := s.transform(ctx, up, count); err != nil {
		return nil, err
	}
	return s.seanet(ctx, up, count)
}

func (s *codecState) upsample(latent []float32, frames int) ([]float32, int) {
	c := s.codec
	return s.convolve(0, codecConv{w: c.upsampleW, cIn: c.outerDim, cOut: c.outerDim, k: c.upsampleK, strd: c.upsampleK / tensor.PairedExtent, transpose: true}, c.outerDim, latent, frames)
}
