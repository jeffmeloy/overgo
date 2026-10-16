// Generation loop: [voice conditioning; text embeddings] prompt prefill,
// then autoregressive frame latents — transformer append, out_norm, EOS
// head, one-step flow decode, latent fed back through input_linear.
package speechsynth

import (
	"context"
	"fmt"

	"overgo/internal/tensor"
)

// GenerateParams declares the loop policy selected by the caller.
// NoiseAt supplies per-step noise from test injection or a caller-owned RNG.
// Session runtime resolves serving defaults before constructing these parameters.
type GenerateParams struct {
	MaxFrames      int
	EOSThreshold   float64
	FramesAfterEOS int // frames decoded after the first EOS crossing
	NoiseAt        func(step int, dst []float32)
}

// LatentBatch: Frames rows of Width normalized frame latents, flat.
type LatentBatch struct {
	// Complete means the reference post-EOS stopping condition was reached.
	// Merely exhausting MaxFrames never establishes completion.
	Complete bool
	Values   []float32
	Frames   int
	Width    int
}

// GenerateLatents runs the seeded loop. voiceCond is [voiceFrames][d]
// already-projected conditioning rows; textIDs are tokenizer output. It
// returns the generated normalized latents and the per-frame EOS logits.
// The reference break rule holds: the frame that trips the post-EOS budget
// is never decoded.
func (m *Model) GenerateLatents(ctx context.Context, voiceCond []float32, voiceFrames int, textIDs []int, p GenerateParams) (LatentBatch, []float64, error) {
	if ctx == nil {
		return LatentBatch{}, nil, fmt.Errorf("speechsynth: generation requires a context")
	}
	if err := context.Cause(ctx); err != nil {
		return LatentBatch{}, nil, err
	}
	d := m.Dims.DModel
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
	return m.generateFromState(ctx, st, p)
}

// GenerateLatentsWithVoice seeds the decode state from an exported
// voice attention state -- the artifact's only voice representation --
// then prompts the text and runs the same seeded loop.
func (m *Model) GenerateLatentsWithVoice(ctx context.Context, voice *VoiceState, textIDs []int, p GenerateParams) (LatentBatch, []float64, error) {
	st, err := m.promptVoice(ctx, voice, textIDs, p)
	if err != nil {
		return LatentBatch{}, nil, err
	}
	return m.generateFromState(ctx, st, p)
}
func (m *Model) promptVoice(ctx context.Context, voice *VoiceState, textIDs []int, p GenerateParams) (*DecodeState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("speechsynth: generation requires a context")
	}
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	if p.MaxFrames <= tensor.FirstOffset || p.NoiseAt == nil {
		return nil, fmt.Errorf("speechsynth: generation needs MaxFrames > 0 and a noise source")
	}
	if len(textIDs) == tensor.FirstOffset {
		return nil, fmt.Errorf("speechsynth: voiced generation needs text")
	}
	st, err := m.VoiceDecodeState(voice, len(textIDs)+p.MaxFrames+tensor.SingletonExtent)
	if err != nil {
		return nil, err
	}
	prompt := make([]float32, len(textIDs)*m.Dims.DModel)
	if err := m.TextEmbedInto(prompt, textIDs); err != nil {
		return nil, err
	}
	m.AppendForward(st, prompt, len(textIDs))
	return st, nil
}

// generateFromState runs the seeded frame loop against a prompted
// decode state.
func (m *Model) generateFromState(ctx context.Context, st *DecodeState, p GenerateParams) (LatentBatch, []float64, error) {
	latents := LatentBatch{Width: m.Dims.LatentDim}
	var eos []float64
	complete, err := m.walkFrames(ctx, st, p, func(frame []float32, logit float64) error {
		latents.Values = append(latents.Values, frame...)
		latents.Frames++
		eos = append(eos, logit)
		return nil
	})
	if err != nil {
		return LatentBatch{}, nil, err
	}
	latents.Complete = complete
	return latents, eos, nil
}

// walkFrames lends each generated frame to emit until emit returns.
// Consumer errors stop generation before the next noise vector is drawn.
func (m *Model) walkFrames(ctx context.Context, st *DecodeState, p GenerateParams, emit func([]float32, float64) error) (bool, error) {
	if err := context.Cause(ctx); err != nil {
		return false, err
	}
	d, latent := m.Dims.DModel, m.Dims.LatentDim

	lat := make([]float32, latent)
	frames, complete := 0, false
	input := make([]float32, d)
	cond := make([]float32, d)
	noise := make([]float32, latent)
	m.LatentInputInto(input, m.BosEmb)
	eosStep := -tensor.SingletonExtent
	for step := tensor.FirstOffset; step < p.MaxFrames; step++ {
		if err := context.Cause(ctx); err != nil {
			return false, err
		}
		m.AppendForward(st, input, tensor.SingletonExtent)
		m.OutNormInto(cond, input, tensor.SingletonExtent)
		logit := m.EOSLogit(cond)
		if logit > p.EOSThreshold && eosStep < tensor.FirstOffset {
			eosStep = step
		}
		if eosStep >= tensor.FirstOffset && step >= eosStep+p.FramesAfterEOS {
			complete = true
			break // reference rule: the breaking frame is not decoded
		}

		p.NoiseAt(step, noise)
		if err := context.Cause(ctx); err != nil {
			return false, err
		}

		m.OneStepLatentInto(lat, cond, noise)
		frames++
		m.LatentInputInto(input, lat)
		if err := emit(lat, logit); err != nil {
			return false, err
		}
	}
	if err := context.Cause(ctx); err != nil {
		return false, err
	}
	if frames == tensor.FirstOffset {
		return false, fmt.Errorf("speechsynth: no frames generated")
	}

	return complete, nil
}
