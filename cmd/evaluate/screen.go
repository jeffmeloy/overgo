package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/vitencoder"
)

// pictureExtensions are the files a render writes a picture to; any other output is a clip, embedded by its frames.
var pictureExtensions = []string{".png", ".jpg", ".jpeg"}

// imageEmbedder gives a picture's pooled embedding: the vision encoder on the card.
type imageEmbedder interface {
	Embed(ctx context.Context, image []byte) ([]float32, error)
}

// screenDraw is one output of a render directory, by the prompt and seed of the request that drew it.
type screenDraw struct {
	prompt, seed, file string
}

// screenRequest is what the screen reads of a draw's request.
type screenRequest struct {
	Prompt string      `json:"prompt"`
	Seed   json.Number `json:"seed"`
}

// readScreenDraws reads a render directory's manifest and its draws, each by its prompt and seed.
func readScreenDraws(directory string) (renderManifest, []screenDraw, error) {
	var held renderManifest
	data, err := os.ReadFile(filepath.Join(directory, renderManifestFile))
	if err != nil {
		return held, nil, err
	}
	if err := json.Unmarshal(data, &held); err != nil {
		return held, nil, fmt.Errorf("evaluate: read the render manifest of %s: %w", directory, err)
	}
	draws := make([]screenDraw, 0, len(held.Draws))
	for _, key := range slices.Sorted(maps.Keys(held.Draws)) {
		drawn := held.Draws[key]
		var request screenRequest
		if err := json.Unmarshal(drawn.Request, &request); err != nil || request.Prompt == "" || request.Seed == "" {
			return held, nil, fmt.Errorf("evaluate: draw %s of %s names no prompt and seed (%v)", key, directory, err)
		}
		draws = append(draws, screenDraw{request.Prompt, request.Seed.String(), filepath.Join(directory, drawn.File)})
	}
	if len(draws) == 0 {
		return held, nil, fmt.Errorf("evaluate: %s holds no draws", directory)
	}
	return held, draws, nil
}

// screenPaths screens the candidate render directory against the reference's: each reference output is matched to the
// candidate's at its prompt and seed, and to the reseeded directory's at its prompt and another seed, which the
// reference path drew with its model, recipe and settings.
func screenPaths(ctx context.Context, encoder imageEmbedder, candidate, reference, reseeded string) (evaluation.Screen, error) {
	referenceManifest, references, err := readScreenDraws(reference)
	if err != nil {
		return evaluation.Screen{}, err
	}
	candidateManifest, candidates, err := readScreenDraws(candidate)
	if err != nil {
		return evaluation.Screen{}, err
	}
	reseededManifest, others, err := readScreenDraws(reseeded)
	if err != nil {
		return evaluation.Screen{}, err
	}
	// A GIF beside an MP4 lies apart by its palette as well as its model, so the three sides agree in format first.
	if format := referenceManifest.clipFormat(); candidateManifest.clipFormat() != format || reseededManifest.clipFormat() != format {
		return evaluation.Screen{}, fmt.Errorf("evaluate: the reference's clips are %q, the candidate's %q and the reseeded directory's %q: a screen compares one format",
			format, candidateManifest.clipFormat(), reseededManifest.clipFormat())
	}
	if reseededManifest.Model != referenceManifest.Model || reseededManifest.Recipe != referenceManifest.Recipe || !maps.Equal(reseededManifest.Settings, referenceManifest.Settings) {
		return evaluation.Screen{}, fmt.Errorf("evaluate: %s was drawn by %s under %s with %v, not by the reference path, %s under %s with %v",
			reseeded, reseededManifest.Model, reseededManifest.Recipe, reseededManifest.Settings, referenceManifest.Model, referenceManifest.Recipe, referenceManifest.Settings)
	}
	cases := make([]evaluation.ScreenCase, 0, len(references))
	for _, drawn := range references {
		same := slices.IndexFunc(candidates, func(other screenDraw) bool { return other.prompt == drawn.prompt && other.seed == drawn.seed })
		another := slices.IndexFunc(others, func(other screenDraw) bool { return other.prompt == drawn.prompt && other.seed != drawn.seed })
		if same < 0 || another < 0 {
			return evaluation.Screen{}, fmt.Errorf("evaluate: the reference drew %q at seed %s; the candidate holds it at that seed: %t, the reseeded directory at another: %t",
				drawn.prompt, drawn.seed, same >= 0, another >= 0)
		}
		own, err := embedOutput(ctx, encoder, drawn.file)
		if err != nil {
			return evaluation.Screen{}, err
		}
		candidateOutput, err := embedOutput(ctx, encoder, candidates[same].file)
		if err != nil {
			return evaluation.Screen{}, err
		}
		otherSeed, err := embedOutput(ctx, encoder, others[another].file)
		if err != nil {
			return evaluation.Screen{}, err
		}
		cases = append(cases, evaluation.ScreenCase{
			Name:      filepath.Base(drawn.file),
			Candidate: cosineDistance(own, candidateOutput),
			Reseeded:  cosineDistance(own, otherSeed),
		})
	}
	return evaluation.ScreenCandidate(cases)
}

// embedOutput is a rendered output's embedding: a picture's own, or the mean of a clip's sampled frames' unit
// embeddings.
func embedOutput(ctx context.Context, encoder imageEmbedder, path string) ([]float32, error) {
	if slices.Contains(pictureExtensions, strings.ToLower(filepath.Ext(path))) {
		picture, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return encoder.Embed(ctx, picture)
	}
	frames, err := media.DecodeVideoFile(ctx, path, os.Getenv("OVERGO_FFMPEG"), frameSampleFPS, frameSampleCount)
	if err != nil || len(frames) == 0 {
		return nil, fmt.Errorf("evaluate: decode the frames of %s: %d frames, %v", path, len(frames), err)
	}
	var mean []float32
	for _, frame := range frames {
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, frame); err != nil {
			return nil, err
		}
		vector, err := encoder.Embed(ctx, encoded.Bytes())
		if err != nil {
			return nil, err
		}
		if mean == nil {
			mean = make([]float32, len(vector))
		}
		var squares float64
		for _, value := range vector {
			squares += float64(value) * float64(value)
		}
		norm := math.Sqrt(squares)
		if len(vector) != len(mean) || norm == 0 {
			return nil, fmt.Errorf("evaluate: a frame of %s embeds to an empty vector or another width", path)
		}
		for index, value := range vector {
			mean[index] += float32(float64(value) / norm / float64(len(frames)))
		}
	}
	return mean, nil
}

// cosineDistance is one minus the cosine of two embeddings; NaN where either is empty, which the screen refuses.
func cosineDistance(a, b []float32) float64 {
	return 1 - evaluation.Cosine(a, b)
}

// runScreen opens the image encoder at encoderPath on the card of device and screens the candidate render directory.
func runScreen(ctx context.Context, encoderPath string, device int, candidate, reference, reseeded string) error {
	encoder, err := vitencoder.Load(encoderPath)
	if err != nil {
		return err
	}
	card, err := encoder.OpenCard(ctx, device)
	if err != nil {
		return err
	}
	screen, err := screenPaths(ctx, card, candidate, reference, reseeded)
	if err != nil {
		return errors.Join(err, card.Close())
	}
	return errors.Join(printScreen(screen), card.Close())
}

// printScreen prints each case the candidate lost and the screen's verdict, and refuses a candidate that failed.
func printScreen(screen evaluation.Screen) error {
	for _, screened := range screen.Cases {
		if screened.Lost() {
			fmt.Printf("screen: %s lost: %.4f from the reference, another seed %.4f\n", screened.Name, screened.Candidate, screened.Reseeded)
		}
	}
	fmt.Printf("screen: the candidate lost %d of %d cases, kept %d, tied %d; chance %.3g at level %.3g\n",
		screen.Lost, len(screen.Cases), screen.Kept, screen.Ties, screen.Tail, screen.Level)
	if !screen.Passed {
		return errors.New("evaluate: the candidate failed the screen: it lies farther from the reference than another seed in more cases than chance allows")
	}
	return nil
}
