package projector

import (
	"context"
	"image"
)

const gemma4ClosedThought = "<|channel>thought\n<channel|>"

// Both artifact architectures share prompt assembly; only their encoders and
// declared assistant prefix differ. The tower oracle has no thought prefix.
type gemmaPromptSource struct {
	width      int
	assistant  string
	image      func(context.Context, image.Image) (Gemma4Output, error)
	audio      func(context.Context, []float32) (Gemma4AudioOutput, error)
	video      func(context.Context, []image.Image) (Gemma4VideoOutput, error)
	sampleRate func() (int, error)
}

func gemma4PromptSource(r *Gemma4Runner) gemmaPromptSource {
	return gemmaPromptSource{
		width: r.spec.Hidden, assistant: gemma4ClosedThought,
		image: r.EncodeImage, audio: r.EncodeAudio, video: r.EncodeVideoFrames,
		sampleRate: func() (int, error) { spec, err := r.AudioSpec(); return spec.SampleRate, err },
	}
}

func gemma4TowerPromptSource(r *Gemma4TowerRunner) gemmaPromptSource {
	return gemmaPromptSource{
		width: r.spec.Vision.ProjectionDim,
		image: func(ctx context.Context, source image.Image) (Gemma4Output, error) {
			output, err := r.EncodeVisionImage(ctx, source)
			return Gemma4Output{Embeddings: output.Embeddings}, err
		},
		audio: func(ctx context.Context, samples []float32) (Gemma4AudioOutput, error) {
			profile, err := NewAudioProjectionProfile(r.preprocess.AudioAttentionRopeFreqBase)
			if err != nil {
				return Gemma4AudioOutput{}, err
			}
			output, err := r.EncodeAudio(ctx, samples, r.spec.Audio.SampleRate, profile)
			return Gemma4AudioOutput{Embeddings: output.Embeddings}, err
		},
		video: func(ctx context.Context, frames []image.Image) (Gemma4VideoOutput, error) {
			output, err := r.EncodeVisionFrames(ctx, frames)
			return Gemma4VideoOutput{Embeddings: output.Embeddings, Frames: output.Frames, TokensPerFrame: output.TokensPerFrame}, err
		},
		sampleRate: func() (int, error) { return r.spec.Audio.SampleRate, nil },
	}
}

func compileGemma4TowerImagePrompt(r *Gemma4TowerRunner) compiledImagePromptProgram {
	return compileGemma4ImageProgram(gemma4TowerPromptSource(r))
}

func compileGemma4TowerMediaPrompt(r *Gemma4TowerRunner) compiledMediaPromptProgram {
	return compileGemma4MediaProgram(gemma4TowerPromptSource(r))
}
