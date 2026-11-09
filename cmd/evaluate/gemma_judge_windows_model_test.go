//go:build windows && modeltest

package main

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestGemmaTwelveOpensAsAJudge opens gemma-4-12B-it from its published directory, unconverted, as the evaluator opens
// a judge: under its active inference and projection recipes, which the owner's answer to gemma-published-activation
// let be activated once its text passed its exact suite and its projection the 27 cases of
// docs/verification/fixtures/gemma-4-12b. It answers the questions the calibration suites ask of their first 12
// pictures and 10 clips, and answers more of them right than wrong by the exact binomial test at calibrationLevel,
// chance taken as one in two, the most any case's choices give. On 2026-11-09 it answered 57 of 67 and 19 of 20 right,
// where Qwen3.8-27B answered 62 of 67 and 19 of 20; how far the two judges agree is second-judge/independent's.
func TestGemmaTwelveOpensAsAJudge(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	root := t.TempDir()
	pictureSuite, clipSuite := calibrationSuites(roots)
	pictures := degradedSuite(t, ctx, ffmpeg, pictureSuite, root, calibrationPictures, nil, nil)
	clips := degradedSuite(t, ctx, ffmpeg, clipSuite, root, calibrationClips, nil, nil)
	directory := roots.ResolveModelPath("gemma-4-12B-it")
	judge := openJudge(t, ctx, roots, root, directory, directory)
	for _, suite := range []evaluation.GeneratedAnswerSuite{pictures, clips} {
		answered := 0
		for _, right := range judge.answer(t, ctx, suite)[unchanged] {
			answered += right
		}
		wrong := len(suite.Cases) - answered
		tail := tensorstats.ExactPairedBinary(uint64(answered), uint64(wrong), 0).Tail
		t.Logf("%s: %d of %d answered right; chance %.3g where right and wrong are as likely", suite.Source, answered, len(suite.Cases), tail)
		if tail > calibrationLevel {
			t.Errorf("gemma-4-12B-it answers %d of the %d questions of %s right, chance %.3g against %.3g", answered, len(suite.Cases), suite.Source, tail, calibrationLevel)
		}
	}
}
