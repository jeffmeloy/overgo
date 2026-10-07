package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
	"overgo/internal/vitencoder"
)

// defaultSceneDistractors is how many clips of a source video with no other scene a
// scene-retrieval suite adds beside the clips that have one, unless a run names another
// number: the gallery a sibling must be found among.
const defaultSceneDistractors = 400

// embeddingRuntime is an image encoder as an evaluation runtime: it embeds a picture of the
// media root named by the digest of its bytes, and generates and scores no text.
type embeddingRuntime struct {
	encoder *vitencoder.Encoder
	root    string
}

// Embed is the vector the encoder gives the picture the name denotes, once the picture's
// bytes are shown to be the digest the name carries.
func (r embeddingRuntime) Embed(_ context.Context, name string) ([]float32, error) {
	path, err := mediaPath(r.root, name)
	if err != nil {
		return nil, err
	}
	picture, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(picture)
	if digest, _, _ := strings.Cut(name, "."); hex.EncodeToString(sum[:]) != digest {
		return nil, errors.New("evaluate: " + name + " is not the picture its digest names")
	}
	return r.encoder.Embed(picture)
}

// Generate refuses: an image encoder writes no text.
func (embeddingRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("evaluate: an image encoder generates no text")
}

// ScoreContinuations refuses: an image encoder scores no text.
func (embeddingRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: an image encoder scores no continuation")
}

// openEmbeddingSession opens the image encoder at the request's path under its active
// image-embedding recipe, embedding the pictures of the request's media root.
func openEmbeddingSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	if request.MediaRoot == "" {
		return nil, fmt.Errorf("evaluate: %s embeds the pictures of a media root, and the request names none", request.Path)
	}
	source, err := mediacapability.Catalog[recipe.TaskImageEmbedding].Resolve(request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no image encoder: %w", request.Path, err)
	}
	return openCapabilitySession(ctx, store, value, source, recipe.TaskImageEmbedding, func() (evaluation.Runtime, func(context.Context) error, error) {
		encoder, err := vitencoder.Load(request.Path)
		return embeddingRuntime{encoder: encoder, root: request.MediaRoot}, func(context.Context) error { return nil }, err
	})
}

// cutFrame is the middle frame of the clip's few sampled frames, as PNG bytes.
func cutFrame(ctx context.Context, clip, ffmpeg string) ([]byte, error) {
	frames, err := media.DecodeVideoFile(ctx, clip, ffmpeg, frameSampleFPS, frameSampleCount)
	if err != nil || len(frames) == 0 {
		return nil, fmt.Errorf("evaluate: cut a frame of %s: %d frames, %v", clip, len(frames), err)
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.Image(frames[len(frames)/frameHalf])); err != nil {
		return nil, err
	}
	return encoded.Bytes(), nil
}

// writeSceneRetrieval cuts a frame from every clip of a directory with a sibling scene and
// from -distractors clips with none, writes each to framesOut named by its digest, and
// writes the retrieval suite over them to out.
func writeSceneRetrieval(ctx context.Context, clips, framesOut, out string, distractors int, seed uint64) error {
	picks, err := evaluation.PickRetrievalClips(clips, distractors, seed)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(framesOut, frameDirectoryMode); err != nil {
		return err
	}
	images := map[string]string{}
	for _, pick := range picks {
		frame, err := cutFrame(ctx, filepath.Join(clips, pick.Clip+clipFileExtension), os.Getenv("OVERGO_FFMPEG"))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(frame)
		name := hex.EncodeToString(sum[:]) + ".png"
		if err := os.WriteFile(filepath.Join(framesOut, name), frame, frameFileMode); err != nil {
			return err
		}
		images[pick.Clip] = name
	}
	suite, err := evaluation.BuildSceneRetrieval(picks, images)
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "scene-retrieval")
}
