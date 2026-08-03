package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/media"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func imageProjectedPrompt(
	ctx context.Context,
	runner *inference.Runner,
	projectorPath, imagePath, question string,
	thinking bool, projectorOptions projector.OpenOptions,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenImageProjectorWithOptions(projectorPath, projectorOptions)
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
	return projectedInputsForPrompt(runner, prompt)
}

func audioProjectedPrompt(
	ctx context.Context,
	runner *inference.Runner,
	projectorPath, audioPath, question string,
	projectorOptions projector.OpenOptions,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	audio, err := projector.OpenAudioProjectorWithOptions(projectorPath, projectorOptions)
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
	return projectedInputsForPrompt(runner, prompt)
}

func videoProjectedPrompt(
	ctx context.Context,
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
	vision, err := projector.OpenVideoProjectorWithOptions(projectorPath, projectorOptions)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open multimodal projector: %w", err)
	}
	defer vision.Close()
	var frames []image.Image
	if videoPath != "" {
		var openErr error
		frames, openErr = decodeVideoFile(ctx, videoPath, ffmpegPath, fps, videoMaxFrames)
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
	return projectedInputsForPrompt(runner, prompt)
}

func decodeVideoFile(ctx context.Context, path, configuredFFmpeg string, fps float64, maxFrames int) ([]image.Image, error) {
	if maxFrames <= 0 || fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return nil, errors.New("video FPS or frame limit is invalid")
	}
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if strings.EqualFold(filepath.Ext(path), ".gif") {
		file, err := os.Open(absolutePath)
		if err != nil {
			return nil, err
		}
		frames, decodeErr := projector.DecodeGIFVideo(file, maxFrames)
		closeErr := file.Close()
		return frames, errors.Join(decodeErr, closeErr)
	}
	ffmpeg, err := resolveFFmpeg(configuredFFmpeg)
	if err != nil {
		return nil, err
	}
	filter := "fps=" + strconv.FormatFloat(fps, 'g', -1, 64)
	command := exec.CommandContext(
		ctx, ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
		"-i", absolutePath, "-vf", filter, "-frames:v", strconv.Itoa(maxFrames),
		"-f", "image2pipe", "-vcodec", "png", "pipe:1",
	)
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr bytes.Buffer
	command.Stderr = &stderr
	if err := command.Start(); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(stdout)
	frames := make([]image.Image, 0, maxFrames)
	for len(frames) < maxFrames {
		frame, decodeErr := png.Decode(reader)
		if errors.Is(decodeErr, io.EOF) || errors.Is(decodeErr, io.ErrUnexpectedEOF) {
			break
		}
		if decodeErr != nil {
			_ = command.Process.Kill()
			_ = command.Wait()
			return nil, decodeErr
		}
		frames = append(frames, frame)
	}
	if err := command.Wait(); err != nil {
		return nil, fmt.Errorf("FFmpeg: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if len(frames) == 0 {
		return nil, errors.New("FFmpeg produced no video frames")
	}
	return frames, nil
}

func resolveFFmpeg(configured string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		if _, err := os.Stat(configured); err != nil {
			return "", fmt.Errorf("FFmpeg executable: %w", err)
		}
		return configured, nil
	}
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		return path, nil
	}
	if runtime.GOOS == "windows" {
		if programFiles := os.Getenv("ProgramFiles"); programFiles != "" {
			candidate := filepath.Join(programFiles, "DownloadHelper CoApp", "ffmpeg.exe")
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
	}
	return "", errors.New("FFmpeg is unavailable; set -ffmpeg or LLAMACPP2GO_FFMPEG")
}

func projectedInputsForPrompt(
	runner *inference.Runner,
	prompt projector.MultimodalPrompt,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	if prompt.EmbeddingWidth != int(runner.Spec().EmbeddingLength) {
		return nil, inference.ProjectedInputs{}, fmt.Errorf(
			"generate: projector width %d differs from model width %d",
			prompt.EmbeddingWidth, runner.Spec().EmbeddingLength,
		)
	}
	imageTokens := len(prompt.EmbeddingTokenIndices)
	if len(prompt.Embeddings) != imageTokens*prompt.EmbeddingWidth {
		return nil, inference.ProjectedInputs{}, errors.New("generate: projector returned invalid embedding indices")
	}
	overrides := make([]inference.EmbeddingOverride, imageTokens)
	for index := range overrides {
		start := index * prompt.EmbeddingWidth
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: prompt.EmbeddingTokenIndices[index],
			Embedding:  prompt.Embeddings[start : start+prompt.EmbeddingWidth],
		}
	}
	projected := inference.ProjectedInputs{EmbeddingOverrides: overrides}
	projected.DeepstackEmbeddings = make([]reference.Value, len(prompt.DeepstackEmbeddings))
	for streamIndex, stream := range prompt.DeepstackEmbeddings {
		if len(stream) != imageTokens*prompt.EmbeddingWidth {
			return nil, inference.ProjectedInputs{}, fmt.Errorf(
				"generate: projector deepstack stream %d has invalid length", streamIndex,
			)
		}
		data := make([]float32, len(prompt.TokenIDs)*prompt.EmbeddingWidth)
		for index, tokenIndex := range prompt.EmbeddingTokenIndices {
			if uint64(tokenIndex) >= uint64(len(prompt.TokenIDs)) {
				return nil, inference.ProjectedInputs{}, errors.New("generate: projector embedding index exceeds prompt")
			}
			source := stream[index*prompt.EmbeddingWidth : (index+1)*prompt.EmbeddingWidth]
			copy(data[int(tokenIndex)*prompt.EmbeddingWidth:], source)
		}
		projected.DeepstackEmbeddings[streamIndex] = reference.Value{
			Shape: tensor.MustShape(uint64(prompt.EmbeddingWidth), uint64(len(prompt.TokenIDs))),
			Data:  data,
		}
	}
	projected.BidirectionalAttentionBlocks = make([]inference.AttentionBlock, len(prompt.AttentionBlocks))
	for index, block := range prompt.AttentionBlocks {
		projected.BidirectionalAttentionBlocks[index] = inference.AttentionBlock{
			Start: block.Start, End: block.End,
		}
	}
	projected.VisualExpertBlocks = make([]inference.AttentionBlock, len(prompt.VisualBlocks))
	for index, block := range prompt.VisualBlocks {
		projected.VisualExpertBlocks[index] = inference.AttentionBlock{Start: block.Start, End: block.End}
	}
	hasMultiAxis := false
	for _, axis := range prompt.MultiAxisPositions {
		hasMultiAxis = hasMultiAxis || len(axis) > 0
	}
	if hasMultiAxis {
		positions := inference.MultiAxisPositions(prompt.MultiAxisPositions)
		projected.MultiAxisPositions = &positions
	}
	return prompt.TokenIDs, projected, nil
}
