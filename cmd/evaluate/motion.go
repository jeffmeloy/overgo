package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path"
	"path/filepath"

	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/vitencoder"
)

// How a clip's frames are read for its motion: every frame of the clips Wan and LiveEdit draw, and their source clips
// at the same rate, so the two compare.
const (
	// motionFrameRate is Wan's and LiveEdit's frames a second.
	motionFrameRate = 16
	// motionFrames is the most frames read of a clip: a Wan clip's 81.
	motionFrames = 81
)

// embedFrames is each frame's embedding, in order.
func embedFrames(ctx context.Context, encoder imageEmbedder, frames []image.Image) ([][]float32, error) {
	vectors := make([][]float32, len(frames))
	for index, frame := range frames {
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, frame); err != nil {
			return nil, err
		}
		vector, err := encoder.Embed(ctx, encoded.Bytes())
		if err != nil {
			return nil, err
		}
		vectors[index] = vector
	}
	return vectors, nil
}

// clipMotion reads the clip at path at motionFrameRate and measures how it moves.
func clipMotion(ctx context.Context, encoder imageEmbedder, ffmpeg, path string) (evaluation.FrameSimilarity, error) {
	frames, err := media.DecodeVideoFile(ctx, path, ffmpeg, motionFrameRate, motionFrames)
	if err != nil {
		return evaluation.FrameSimilarity{}, fmt.Errorf("evaluate: read the frames of %s: %w", path, err)
	}
	vectors, err := embedFrames(ctx, encoder, frames)
	if err != nil {
		return evaluation.FrameSimilarity{}, err
	}
	measured, err := evaluation.NewFrameSimilarity(vectors)
	if err != nil {
		return evaluation.FrameSimilarity{}, fmt.Errorf("evaluate: %s: %w", path, err)
	}
	return measured, nil
}

// measureMotion measures how each clip a suite asks about moves, read from clips, and how the source clip each describes
// moves where sources holds one by its name (MotionKey's last element and the clip extension); a caption composed for
// the suite describes no source.
func measureMotion(ctx context.Context, encoder imageEmbedder, encoderName string, suite *evaluation.GeneratedAnswerSuite, clips, sources string) error {
	ffmpeg, err := media.FFmpeg()
	if err != nil {
		return err
	}
	motion := evaluation.ClipMotion{Encoder: encoderName, Clips: map[string]evaluation.FrameSimilarity{}, Sources: map[string]evaluation.FrameSimilarity{}}
	for _, testCase := range suite.Cases {
		kind, name, _, framed := evaluation.ParseMediaPrompt(testCase.Prompt)
		key := testCase.MotionKey()
		if _, measured := motion.Clips[key]; !framed || kind != evaluation.MediaVideo || measured {
			continue
		}
		if motion.Clips[key], err = clipMotion(ctx, encoder, ffmpeg, filepath.Join(clips, name)); err != nil {
			return err
		}
		if sources == "" {
			continue
		}
		source := filepath.Join(sources, path.Base(key)+clipFileExtension)
		if _, err := os.Stat(source); errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return err
		}
		if motion.Sources[key], err = clipMotion(ctx, encoder, ffmpeg, source); err != nil {
			return err
		}
	}
	if len(motion.Clips) == 0 {
		return errors.New("evaluate: the suite asks about no clip")
	}
	suite.Motion = &motion
	return nil
}

// writeMotion measures the clips of the suite at suitePath with the image encoder at encoderPath on the card of device,
// and writes the suite with their motion to out.
func writeMotion(ctx context.Context, suitePath, clips, sources, encoderPath string, device int, out string) error {
	var suite evaluation.GeneratedAnswerSuite
	if err := jsonfile.DecodeStrict(suitePath, &suite); err != nil {
		return err
	}
	encoder, err := vitencoder.Load(encoderPath)
	if err != nil {
		return err
	}
	card, err := encoder.OpenCard(ctx, device)
	if err != nil {
		return err
	}
	if err := measureMotion(ctx, card, filepath.Base(encoderPath), &suite, clips, sources); err != nil {
		return errors.Join(err, card.Close())
	}
	if err := card.Close(); err != nil {
		return err
	}
	for _, metric := range suite.Motion.Metrics() {
		fmt.Printf("motion: %s %.4f\n", metric.Name, metric.Value)
	}
	return writeSuite(out, suite, len(suite.Cases), "clip motion")
}
