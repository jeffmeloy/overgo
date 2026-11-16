package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

// pairedDraw is a request both arms of a reading drew: the reference's draw of it and the preset's, and its prompt.
type pairedDraw struct {
	reference, preset renderedMedia
	prompt            string
}

// pairedDraws pairs each of the reference's draws with the preset's draw of its request, the preset aside, by the
// reference's key: two renders that are not one request but the preset are refused (comparableArms).
func pairedDraws(reference, preset renderManifest) (map[string]pairedDraw, error) {
	if err := comparableArms(reference, preset); err != nil {
		return nil, err
	}
	asked, err := unpresetDraws(reference)
	if err != nil {
		return nil, err
	}
	guided, err := unpresetDraws(preset)
	if err != nil {
		return nil, err
	}
	pairs := map[string]pairedDraw{}
	for request, key := range asked {
		var fields struct {
			Prompt string `json:"prompt"`
		}
		if err := json.Unmarshal([]byte(request), &fields); err != nil || fields.Prompt == "" {
			return nil, errors.Join(err, fmt.Errorf("evaluate: draw %s asked for no prompt a pair can be read against", key))
		}
		pairs[key] = pairedDraw{reference: reference.Draws[key], preset: preset.Draws[guided[request]], prompt: fields.Prompt}
	}
	return pairs, nil
}

// readPicture decodes the picture a render directory holds as file.
func readPicture(directory, file string) (image.Image, error) {
	path, err := mediaPath(directory, file)
	if err != nil {
		return nil, err
	}
	opened, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	picture, _, err := image.Decode(opened)
	return picture, errors.Join(err, opened.Close())
}

// writePairwise composes the pairwise reading of the preset drawn into presetDirectory against its reference drawn
// into referenceDirectory (pairwise-reading/do): each request both drew, its two pictures side by side in each order,
// written to mediaOut as evaluation.PairwiseMedium names them, and the suite that asks judge which better shows its
// prompt, written to suiteOut.
func writePairwise(referenceDirectory, presetDirectory, mediaOut, suiteOut, judge string) error {
	reference, err := readRender(referenceDirectory)
	if err != nil {
		return err
	}
	preset, err := readRender(presetDirectory)
	if err != nil {
		return err
	}
	if reference.Format != "" || preset.Format != "" {
		return errors.New("evaluate: a pairwise reading sets pictures side by side, and these renders drew clips")
	}
	pairs, err := pairedDraws(reference, preset)
	if err != nil {
		return err
	}
	referenceArm, err := drawnArm(reference)
	if err != nil {
		return err
	}
	presetArm, err := drawnArm(preset)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(mediaOut, renderDirectoryMode); err != nil {
		return err
	}
	prompts := map[string]string{}
	for medium, pair := range pairs {
		referencePicture, err := readPicture(referenceDirectory, pair.reference.File)
		if err != nil {
			return err
		}
		presetPicture, err := readPicture(presetDirectory, pair.preset.File)
		if err != nil {
			return err
		}
		for side, order := range map[string][2]image.Image{"A": {presetPicture, referencePicture}, "B": {referencePicture, presetPicture}} {
			composite, err := evaluation.ComposePair(order[0], order[1])
			if err != nil {
				return fmt.Errorf("evaluate: draw %s: %w", medium, err)
			}
			if err := writePNG(filepath.Join(mediaOut, evaluation.PairwiseMedium(medium, side)), composite); err != nil {
				return err
			}
		}
		prompts[medium] = pair.prompt
	}
	suite, err := evaluation.BuildPairwise(referenceArm, presetArm, prompts, judge)
	if err != nil {
		return err
	}
	return jsonfile.Write(suiteOut, suite, renderFileMode)
}

// writePNG writes picture to path as a PNG.
func writePNG(path string, picture image.Image) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, renderFileMode)
	if err != nil {
		return err
	}
	return errors.Join(png.Encode(file, picture), file.Close())
}

// readPairwiseVerdict prints the pairwise verdict of the preset drawn into presetDirectory from the judged evaluation
// record of its pairwise suite: each case right where the judge chose the preset's picture (evaluation.ReadPairwise).
func readPairwiseVerdict(ctx context.Context, repository, presetDirectory, record string, said io.Writer) error {
	preset, err := readRender(presetDirectory)
	if err != nil {
		return err
	}
	id, err := artifact.ParseID(record)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return err
	}
	defer store.Close()
	chose, found := evaluation.CaseOutcomes(ctx, store, id)
	if !found {
		return fmt.Errorf("evaluate: no case outcomes can be read from evaluation %s", id)
	}
	media, found := evaluation.CaseMedia(ctx, store, id)
	if !found {
		return fmt.Errorf("evaluate: what each case asks about cannot be read from evaluation %s", id)
	}
	verdict, err := evaluation.ReadPairwise(preset.Preset, chose, media)
	if err != nil {
		return err
	}
	encoded, err := json.Marshal(verdict)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(said, "%s\n", encoded)
	return err
}
