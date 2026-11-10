//go:build windows && modeltest

package main

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/processmeasure"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// judgeSpeedQuestions is how many of the calibration pictures' questions the two judges are timed on.
const judgeSpeedQuestions = 24

// gemmaHostProbabilities are gemma-4-12B-it's probabilities of the right answer to the judgeSpeedQuestions questions as
// the host-built prompt path read them on 2026-11-10, before its picture prompts were read on the card; the card's
// reading holds each within gemmaProbabilityTolerance.
var gemmaHostProbabilities = map[string]float64{
	"unchanged/tifa-v1.0/image-questions/partiprompt_686/0": 1.0000, "unchanged/tifa-v1.0/image-questions/partiprompt_686/1": 1.0000,
	"unchanged/tifa-v1.0/image-questions/partiprompt_686/2": 0.0042, "unchanged/tifa-v1.0/image-questions/partiprompt_686/3": 0.9928,
	"unchanged/tifa-v1.0/image-questions/partiprompt_686/4": 0.0000, "unchanged/tifa-v1.0/image-questions/partiprompt_686/5": 0.0077,
	"unchanged/tifa-v1.0/image-questions/partiprompt_1003/0": 1.0000, "unchanged/tifa-v1.0/image-questions/partiprompt_1003/1": 0.9999,
	"unchanged/tifa-v1.0/image-questions/partiprompt_1003/2": 0.9996, "unchanged/tifa-v1.0/image-questions/partiprompt_1003/3": 0.9997,
	"unchanged/tifa-v1.0/image-questions/partiprompt_1003/4": 1.0000, "unchanged/tifa-v1.0/image-questions/coco_376292/0": 1.0000,
	"unchanged/tifa-v1.0/image-questions/coco_376292/1": 1.0000, "unchanged/tifa-v1.0/image-questions/coco_376292/2": 1.0000,
	"unchanged/tifa-v1.0/image-questions/coco_376292/3": 1.0000, "unchanged/tifa-v1.0/image-questions/coco_376292/4": 1.0000,
	"unchanged/tifa-v1.0/image-questions/coco_376292/5": 1.0000, "unchanged/tifa-v1.0/image-questions/coco_376292/6": 1.0000,
	"unchanged/tifa-v1.0/image-questions/partiprompt_993/0": 0.0032, "unchanged/tifa-v1.0/image-questions/partiprompt_993/1": 1.0000,
	"unchanged/tifa-v1.0/image-questions/partiprompt_993/2": 1.0000, "unchanged/tifa-v1.0/image-questions/coco_657311/0": 1.0000,
	"unchanged/tifa-v1.0/image-questions/coco_657311/1": 0.9999, "unchanged/tifa-v1.0/image-questions/coco_657311/2": 1.0000,
}

// gemmaProbabilityTolerance is how far the card's reading of a question's probability may stand from the host's.
const gemmaProbabilityTolerance = 0.02

// TestGemmaTwelveJudgesNoSlowerThanTheLargerJudge times Qwen3.8-27B and gemma-4-12B-it judging the same
// judgeSpeedQuestions questions about the calibration suite's Krea pictures, each a choice whose probability is read,
// and says where each judge's time went: the picture read and decoded, the projector's encode, the prompt read, the
// answer written, and a text prompt as long, read by the same judge. On 2026-11-10 gemma took 0.517 s a question to
// Qwen's 0.333 s, 501 ms of it in a prompt read on the host-built path where its text prompts take 78 ms. Its picture
// prompts read on the card, gemma judges a question no slower than Qwen, and every probability is the host's.
func TestGemmaTwelveJudgesNoSlowerThanTheLargerJudge(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	root := t.TempDir()
	pictures, _ := calibrationSuites(roots)
	suite := withLetteredChoices(degradedSuite(t, ctx, ffmpeg, pictures, root, judgeSpeedQuestions, nil, nil))
	suite.Cases = suite.Cases[:min(len(suite.Cases), judgeSpeedQuestions)]
	qwen := filepath.Join(roots.ResolveModelPath("Qwen3.8-27B-UD-Q6_K"), "Qwen3.8-27B-UD-Q6_K.gguf")
	gemma := roots.ResolveModelPath("gemma-4-12B-it")
	judges := []struct{ name, path, projector string }{
		{"Qwen3.8-27B", qwen, filepath.Join(filepath.Dir(qwen), "mmproj-BF16.gguf")},
		{"gemma-4-12B-it", gemma, gemma},
	}
	took := make([]float64, len(judges))
	for at, judge := range judges {
		t.Run(judge.name, func(t *testing.T) {
			opened := openJudge(t, ctx, roots, root, judge.path, judge.projector)
			staged, stages := processmeasure.WithStages(ctx)
			started := time.Now()
			answers := opened.judged(t, staged, suite)
			took[at] = time.Since(started).Seconds() / float64(len(suite.Cases))
			tokens := 0
			for _, testCase := range suite.Cases {
				tokens += answers[testCase.Name].prompt
				if host, pinned := gemmaHostProbabilities[testCase.Name]; judge.name == "gemma-4-12B-it" &&
					(!pinned || math.Abs(answers[testCase.Name].probability-host) > gemmaProbabilityTolerance) {
					t.Errorf("gemma's probability of the right answer to %s is %.4f, the host path's %.4f (pinned %t)", testCase.Name, answers[testCase.Name].probability, host, pinned)
				}
			}
			parts := testcheck.Must(stages.Parts())(t)
			var said []string
			var prompt time.Duration
			for _, part := range parts {
				if part.Name == "prompt" {
					prompt = part.Wall
				}
				if part.Wall >= time.Millisecond {
					said = append(said, part.Name+" "+(part.Wall/time.Duration(len(suite.Cases))).Round(time.Millisecond).String())
				}
			}
			t.Logf("%s: %.3f s a question over %d questions, %d prompt tokens a question read at %.0f tok/s; a question's %s", judge.name, took[at],
				len(suite.Cases), tokens/len(suite.Cases), float64(tokens)/prompt.Seconds(), strings.Join(said, ", "))
			// The same judge reads a prompt of text alone as long as a question, so the picture's own cost is told from
			// the model's.
			text := strings.Repeat("The picture shows a quiet street at dusk. ", tokens/len(suite.Cases)/9)
			var read time.Duration
			var textTokens int
			for range 4 {
				began := time.Now()
				_, _, err := opened.session.runner.Generate(ctx, text, inference.GenerateOptions{MaxNewTokens: 1,
					OnPromptEvaluated: func(evaluated inference.PromptEvaluation) {
						read, textTokens = read+time.Since(began), evaluated.Tokens
					}})
				testcheck.NoError(t, err)
			}
			t.Logf("%s: a text prompt of %d tokens read in %s", judge.name, textTokens, (read / 4).Round(time.Millisecond))
		})
	}
	if t.Failed() {
		return
	}
	if took[1] > took[0] {
		t.Errorf("gemma-4-12B-it took %.3f s a question, Qwen3.8-27B %.3f s; want gemma no slower", took[1], took[0])
	}
}
