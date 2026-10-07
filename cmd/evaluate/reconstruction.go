package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

const (
	// defaultReconstructionFrames is how many frames of each clip a reconstruction suite
	// gives the autoencoder unless a run names another number: 21 is five temporal strides
	// of four and the frame that opens them, the length the video editor's sources are cut to.
	defaultReconstructionFrames = 21
	// reconstructionFPS is the rate a reconstruction suite cuts its clips at, and
	// defaultReconstructionWidth by defaultReconstructionHeight the size unless a run names
	// another: the 480p frame the installed video autoencoder's generator draws at. An
	// autoencoder that works at one size, as a representation autoencoder does at 256 by
	// 256, is given a suite cut to that size.
	reconstructionFPS           = 16
	defaultReconstructionWidth  = 832
	defaultReconstructionHeight = 480
)

// reconstructionRuntime is an autoencoder as an evaluation runtime: it passes a clip of the
// media root through the autoencoder and back, and generates and scores no text.
type reconstructionRuntime struct {
	autoencoder mediacapability.Autoencoder
	root        string
	ffmpeg      string
}

// Reconstruct cuts the clip the case names as the suite says, once the clip's bytes are
// shown to be the digest the case carries, and returns it as it went into the autoencoder
// and as it came back.
func (r reconstructionRuntime) Reconstruct(ctx context.Context, clip evaluation.ReconstructionCase, cut evaluation.ClipCut) (evaluation.RoundTrip, error) {
	path, err := mediaPath(r.root, clip.Clip)
	if err != nil {
		return evaluation.RoundTrip{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return evaluation.RoundTrip{}, err
	}
	if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != clip.SHA256 {
		return evaluation.RoundTrip{}, errors.New("evaluate: " + clip.Clip + " is not the clip its digest names")
	}
	frames, err := media.DecodeVideoFileAt(ctx, path, r.ffmpeg, float64(cut.FPS), cut.Frames, cut.Width, cut.Height)
	if err != nil || len(frames) != cut.Frames {
		return evaluation.RoundTrip{}, fmt.Errorf("evaluate: %s gave %d of the %d frames asked: %w", clip.Clip, len(frames), cut.Frames, err)
	}
	source, err := latentvideo.SourceVideoFromFrames(frames)
	if err != nil {
		return evaluation.RoundTrip{}, err
	}
	output, latent, err := r.autoencoder.RoundTrip(ctx, source)
	if err != nil {
		return evaluation.RoundTrip{}, err
	}
	return evaluation.RoundTrip{
		Source: source.Pixels, Output: output.Pixels,
		LatentValues: latent.Channels * latent.Frames * latent.Height * latent.Width,
	}, nil
}

// Generate refuses: an autoencoder writes no text.
func (reconstructionRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("evaluate: an autoencoder generates no text")
}

// ScoreContinuations refuses: an autoencoder scores no text.
func (reconstructionRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: an autoencoder scores no continuation")
}

// openReconstructionSession opens the autoencoder at the request's path under its active
// reconstruction recipe, taking the clips of the request's media root.
func openReconstructionSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	if request.MediaRoot == "" {
		return nil, fmt.Errorf("evaluate: %s reconstructs the clips of a media root, and the request names none", request.Path)
	}
	source, err := mediacapability.Catalog[recipe.TaskReconstruction].Resolve(request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no autoencoder: %w", request.Path, err)
	}
	return openCapabilitySession(ctx, store, value, source, recipe.TaskReconstruction, func() (evaluation.Runtime, func(context.Context) error, error) {
		autoencoder, err := mediacapability.OpenAutoencoder(request.Path)
		if err != nil {
			return nil, nil, err
		}
		return reconstructionRuntime{autoencoder: autoencoder, root: request.MediaRoot, ffmpeg: os.Getenv("OVERGO_FFMPEG")}, autoencoder.Close, nil
	})
}

// writeClipReconstruction writes the reconstruction suite of cases clips of a directory,
// picked by seed and each cut to frames frames of width by height, to out.
func writeClipReconstruction(clips, out string, cases, frames, width, height int, seed uint64) error {
	suite, err := evaluation.BuildClipReconstruction(clips, cases, seed, evaluation.ClipCut{
		Frames: frames, FPS: reconstructionFPS, Width: width, Height: height,
	})
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "reconstruction")
}
