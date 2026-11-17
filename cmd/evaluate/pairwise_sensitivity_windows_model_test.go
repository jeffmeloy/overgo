//go:build windows && modeltest

package main

import (
	"encoding/json"
	"image"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestThePairwiseJudgeSeesAPictureOfAnotherPrompt shows the pairwise reading to be an instrument before any preset is
// read by it (pairwise-reading/sensitivity). MOSS-Image-SFT's reference render of the held-out suite, drawn at a
// person's size, is read pairwise against three arms of its own pictures, each written as a preset's render of the
// same requests: each picture swapped for the next draw's, another prompt's; each blurred as the judge's calibration
// blurs a picture (blurred), which the questions were measured to see; and each unchanged. The served judge,
// Qwen3.8-27B, prefers the reference's picture to another prompt's and to a blurred one, each arm refused on a loss
// beyond chance by the grouped sign test the reading uses, and the unchanged arm shows neither a loss nor a gain beyond
// chance. One suite asks all three arms, each arm's composites under a folder of its own, and the judge's answers are
// kept in a store of the test's own.
func TestThePairwiseJudgeSeesAPictureOfAnotherPrompt(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	referenceDirectory := filepath.Join(roots.Datasets, "tifa_v1.0", "moss-image-sft-person")
	reference := testcheck.Must(readRender(referenceDirectory))(t)
	keys := slices.Sorted(maps.Keys(reference.Draws))
	pictures := make([]image.Image, len(keys))
	for at, key := range keys {
		pictures[at] = testcheck.Must(readPicture(referenceDirectory, reference.Draws[key].File))(t)
	}
	arms := []struct {
		name    string
		picture func(at int) image.Image
		worse   bool
	}{
		{name: "swapped", picture: func(at int) image.Image { return pictures[(at+1)%len(pictures)] }, worse: true},
		{name: "blurred", picture: func(at int) image.Image { return blurred(pictures[at]) }, worse: true},
		{name: unchanged, picture: func(at int) image.Image { return pictures[at] }},
	}
	root := t.TempDir()
	var suite evaluation.GeneratedAnswerSuite
	for _, arm := range arms {
		directory := t.TempDir()
		drawn := reference
		drawn.Preset, drawn.Draws = arm.name, map[string]renderedMedia{}
		for at, key := range keys {
			draw := reference.Draws[key]
			draw.File = key + ".png"
			testcheck.NoError(t, writePNG(filepath.Join(directory, draw.File), arm.picture(at)))
			drawn.Draws[key] = draw
		}
		testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), drawn, renderFileMode))
		path := filepath.Join(t.TempDir(), "pairwise.json")
		testcheck.NoError(t, writePairwise(referenceDirectory, directory, filepath.Join(root, arm.name), path))
		var asked evaluation.GeneratedAnswerSuite
		testcheck.NoError(t, jsonfile.DecodeStrict(path, &asked))
		if suite.Cases == nil {
			suite = asked
			suite.Cases = []evaluation.GeneratedAnswerCase{}
		}
		for _, testCase := range asked.Cases {
			_, named, _, _ := evaluation.ParseMediaPrompt(testCase.Prompt)
			testCase.Prompt = strings.Replace(testCase.Prompt, "name="+named+">", "name="+arm.name+"/"+named+">", 1)
			testCase.Name, testCase.Group = arm.name+"/"+testCase.Name, arm.name+"/"+testCase.Group
			suite.Cases = append(suite.Cases, testCase)
		}
	}

	judge := openCalibration(t, ctx, roots, root)
	compiled := testcheck.Must(judge.session.campaign.CompileSuite(ctx, testcheck.Must(json.Marshal(suite))(t)))(t)
	result := testcheck.Must(judge.session.campaign.Evaluate(ctx, compiled))(t)
	outcomes, read := evaluation.CaseOutcomes(ctx, judge.answers, result.Evaluation)
	if !read || len(outcomes) != len(suite.Cases) {
		t.Fatalf("the judge's %d answers to the %d cases cannot be read", len(outcomes), len(suite.Cases))
	}
	for _, arm := range arms {
		chose, media := map[string]bool{}, map[string]string{}
		for _, testCase := range suite.Cases {
			if name, ours := strings.CutPrefix(testCase.Name, arm.name+"/"); ours {
				chose[name], media[name] = outcomes[testCase.Name], testCase.Group
			}
		}
		verdict := testcheck.Must(evaluation.ReadPairwise(arm.name, chose, media))(t)
		t.Logf("%s against MOSS's reference: %s; the gain's tail %.3g", arm.name, verdict.Says, verdict.Tail)
		if arm.worse == verdict.Admitted || !arm.worse && verdict.Tail <= calibrationLevel {
			t.Errorf("the pairwise judge reads %s pictures as %s; want a loss beyond chance for a known worse picture, and neither a loss nor a gain for the same one",
				arm.name, verdict.Says)
		}
	}
}
