package main

import (
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
	return inference.ProjectedInputsForPrompt(runner, prompt)
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
	var samples []float32
	switch strings.ToLower(filepath.Ext(audioPath)) {
	case ".f32":
		samples, err = media.DecodeFloat32LE(data)
	case ".wav":
		var sampleRate int
		samples, sampleRate, err = media.DecodeWAV(data)
		if err == nil && sampleRate != 16000 {
			err = fmt.Errorf("sample rate %d Hz; want 16000 Hz", sampleRate)
		}
	default:
		err = errors.New("audio input must use .wav or .f32")
	}
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode audio: %w", err)
	}
	prompt, err := audio.BuildAudioPrompt(ctx, runner, samples, "", question)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode audio: %w", err)
	}
	return inference.ProjectedInputsForPrompt(runner, prompt)
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
	return inference.ProjectedInputsForPrompt(runner, prompt)
}
