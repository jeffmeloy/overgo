package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/projector"
	"overgo/internal/tokenizer"
)

func imageProjectedPrompt(
	ctx context.Context,
	store artifact.Reader,
	runner *inference.Runner,
	projectorPath, imagePath, question string,
	thinking bool, projectorOptions projector.OpenOptions,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenActiveSession(ctx, store, runner.ModelID(), projectorPath, projectorOptions)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open multimodal projector: %w", err)
	}
	defer vision.Close()
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open image: %w", err)
	}
	input, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode image: %w", err)
	}
	prompt, err := vision.BuildImagePrompt(ctx, runner, input, "", question, thinking)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode image: %w", err)
	}
	return inference.CompileProjectedInputs(prompt, runner.Spec().EmbeddingLength)
}

func audioProjectedPrompt(
	ctx context.Context,
	store artifact.Reader,
	runner *inference.Runner,
	projectorPath, audioPath, question string,
	projectorOptions projector.OpenOptions,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	audio, err := projector.OpenActiveSession(ctx, store, runner.ModelID(), projectorPath, projectorOptions)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open audio projector: %w", err)
	}
	defer audio.Close()
	data, err := os.ReadFile(audioPath)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: read audio: %w", err)
	}
	var want int
	if strings.EqualFold(filepath.Ext(audioPath), ".wav") {
		want, err = projector.AudioSampleRate(audio)
		if err != nil {
			return nil, inference.ProjectedInputs{}, err
		}
	}
	samples, err := decodeProjectedAudio(ctx, data, filepath.Ext(audioPath), want)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode audio: %w", err)
	}
	prompt, err := audio.BuildAudioPrompt(ctx, runner, samples, "", question)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode audio: %w", err)
	}
	return inference.CompileProjectedInputs(prompt, runner.Spec().EmbeddingLength)
}

func decodeProjectedAudio(ctx context.Context, data []byte, extension string, sampleRate int) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("audio input requires a context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch strings.ToLower(extension) {
	case ".f32":
		samples, err := media.DecodeFloat32LE(data)
		if contextErr := ctx.Err(); contextErr != nil {
			return nil, contextErr
		}
		return samples, err
	case ".wav":
		if !bytes.HasPrefix(data, []byte("RIFF")) {
			return nil, errors.New(".wav input must contain RIFF/WAVE audio")
		}
		// Every supported WAV scalar consumes at least one encoded byte.
		audio, _, err := media.DecodeAudio(ctx, data, uint64(len(data)))
		if err != nil {
			return nil, err
		}
		if audio.Format.Channels != 1 {
			return nil, errors.New("audio projector requires mono samples")
		}
		if sampleRate <= 0 || audio.Format.SampleRate != uint64(sampleRate) {
			return nil, fmt.Errorf("sample rate %d Hz; want %d Hz", audio.Format.SampleRate, sampleRate)
		}
		return audio.Samples, nil
	default:
		return nil, errors.New("audio input must use .wav or .f32")
	}
}

func videoProjectedPrompt(
	ctx context.Context,
	store artifact.Reader,
	runner *inference.Runner,
	projectorPath string,
	framePaths []string,
	videoPath string,
	videoMaxFrames int,
	ffmpegPath string,
	question string,
	fps float64,
	thinking bool, projectorOptions projector.OpenOptions,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenActiveSession(ctx, store, runner.ModelID(), projectorPath, projectorOptions)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open multimodal projector: %w", err)
	}
	defer vision.Close()
	var frames []image.Image
	if videoPath != "" {
		var openErr error
		frames, openErr = media.DecodeVideoFile(ctx, videoPath, ffmpegPath, fps, videoMaxFrames)
		if openErr != nil {
			return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode video: %w", openErr)
		}
	} else {
		frames = make([]image.Image, len(framePaths))
		for index, path := range framePaths {
			file, openErr := os.Open(path)
			if openErr != nil {
				return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open video frame %d: %w", index, openErr)
			}
			frames[index], _, openErr = image.Decode(file)
			_ = file.Close()
			if openErr != nil {
				return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode video frame %d: %w", index, openErr)
			}
		}
	}
	prompt, err := vision.BuildVideoPrompt(ctx, runner, frames, "", question, fps, thinking)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode video: %w", err)
	}
	return inference.CompileProjectedInputs(prompt, runner.Spec().EmbeddingLength)
}
