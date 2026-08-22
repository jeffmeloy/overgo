// Generation loop: [voice conditioning; text embeddings] prompt prefill,
// then autoregressive frame latents — transformer append, out_norm, EOS
// head, one-step flow decode, latent fed back through input_linear.
package speechsynth

import (
	"fmt"

	"overgo/internal/tensor"
)

// GenerateParams: explicit loop policy. NoiseAt supplies the per-step noise
// (test injection or a caller-owned RNG); serving-policy defaults belong to
// a later slice.
type GenerateParams struct {
	MaxFrames      int
	EOSThreshold   float64
	FramesAfterEOS int // frames decoded after the first EOS crossing
	NoiseAt        func(step int, dst []float32)
}

// LatentBatch: Frames rows of Width normalized frame latents, flat.
type LatentBatch struct {
	Values []float32
	Frames int
	Width  int
}

// GenerateLatents runs the seeded loop. voiceCond is [voiceFrames][d]
// already-projected conditioning rows; textIDs are tokenizer output. It
// returns the generated normalized latents and the per-frame EOS logits.
// The reference break rule holds: the frame that trips the post-EOS budget
// is never decoded.
func (m *Model) GenerateLatents(voiceCond []float32, voiceFrames int, textIDs []int, p GenerateParams) (LatentBatch, []float64, error) {
	d, latent := m.Dims.DModel, m.Dims.LatentDim
	if len(voiceCond) != voiceFrames*d {
		return LatentBatch{}, nil, fmt.Errorf("speechsynth: voice conditioning %d values != %d rows x %d", len(voiceCond), voiceFrames, d)
	}
	if p.MaxFrames <= tensor.FirstOffset || p.NoiseAt == nil {
		return LatentBatch{}, nil, fmt.Errorf("speechsynth: generation needs MaxFrames > 0 and a noise source")
	}
	st := m.NewDecodeState(voiceFrames + len(textIDs) + p.MaxFrames + tensor.SingletonExtent)

	// Prompt phases: outputs beyond the last row are discarded.
	prompt := make([]float32, (voiceFrames+len(textIDs))*d)
	copy(prompt, voiceCond)
	if err := m.TextEmbedInto(prompt[voiceFrames*d:], textIDs); err != nil {
		return LatentBatch{}, nil, err
	}
	m.AppendForward(st, prompt, voiceFrames+len(textIDs))

	latents := LatentBatch{Values: make([]float32, p.MaxFrames*latent), Width: latent}
	eos := make([]float64, tensor.FirstOffset, p.MaxFrames)
	input := make([]float32, d)
	cond := make([]float32, d)
	noise := make([]float32, latent)
	m.LatentInputInto(input, m.BosEmb)
	eosStep := -tensor.SingletonExtent
	for step := tensor.FirstOffset; step < p.MaxFrames; step++ {
		m.AppendForward(st, input, tensor.SingletonExtent)
		m.OutNormInto(cond, input, tensor.SingletonExtent)
		logit := m.EOSLogit(cond)
		if logit > p.EOSThreshold && eosStep < tensor.FirstOffset {
			eosStep = step
		}
		if eosStep >= tensor.FirstOffset && step >= eosStep+p.FramesAfterEOS {
			break // reference rule: the breaking frame is not decoded
		}
		eos = append(eos, logit)
		p.NoiseAt(step, noise)
		lat := latents.Values[latents.Frames*latent : (latents.Frames+tensor.SingletonExtent)*latent]
		m.OneStepLatentInto(lat, cond, noise)
		latents.Frames++
		m.LatentInputInto(input, lat)
	}
	if latents.Frames == tensor.FirstOffset {
		return LatentBatch{}, nil, fmt.Errorf("speechsynth: no frames generated")
	}
	latents.Values = latents.Values[:latents.Frames*latent]
	return latents, eos, nil
}
