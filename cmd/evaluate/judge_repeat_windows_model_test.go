//go:build windows && modeltest

package main

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The judge's repeat, declared on 2026-11-08 before the judge was asked twice. A paired comparison reads a case's win
// or loss from the judge's letters and the soft share from its probability of the right letter; a soft share compared
// between two candidates moves by hundredths, so the judge's own drift from run to run is held a tenth of that.
const (
	// judgeRepeatCases is how many of each frozen suite's cases the judge is asked, twice.
	judgeRepeatCases = 32
	// judgeRepeatTolerance is how far the judge's probability of a case's right letter, and the frozen set's soft
	// share, may move between two runs at one commit.
	judgeRepeatTolerance = 1e-3
)

// TestTheJudgeAnswersTheSameTwice holds the judge, Qwen3.8-27B through its projector, to a repeatable instrument: a
// paired comparison counts wins and losses case by case (skill.md), and a judge that answers differently from run to run
// makes wins and losses no change made. Some kernels on its path add with atomicAdd, whose order can vary. The judge is
// asked the first 32 questions of MOSS-Image-SFT's TIFA pictures and of the Wan student's clips, each picture and clip
// verified as the render that drew it, twice in one process; it gives the same letter each time, and its probability of
// each right letter and the set's soft share move by no more than judgeRepeatTolerance.
func TestTheJudgeAnswersTheSameTwice(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	judgeDirectory := roots.ResolveModelPath("Qwen3.8-27B-UD-Q6_K")
	judgePath, projectorPath := filepath.Join(judgeDirectory, "Qwen3.8-27B-UD-Q6_K.gguf"), filepath.Join(judgeDirectory, "mmproj-BF16.gguf")
	frozen := []struct{ suite, media string }{
		{filepath.Join(roots.Datasets, "tifa_v1.0", "moss-image-sft-suite.json"), filepath.Join(roots.Datasets, "tifa_v1.0", "moss-image-sft")},
		{filepath.Join(roots.Datasets, "VidGen", "self-forcing-generated-suite.json"), filepath.Join(roots.Datasets, "VidGen", "self-forcing-generated")},
	}
	for _, path := range []string{judgePath, projectorPath, frozen[0].suite, frozen[1].suite} {
		if _, err := os.Stat(path); err != nil {
			testskip.NotApplicable(t, "the judge or a frozen suite is not on this machine: "+path)
		}
	}
	store := testcheck.Must(overgodb.OpenContext(ctx, roots.Store))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	reserve, bind := modelcli.ModelReserve(ctx)
	judgeID, _, err := modelcli.ResolveModel(ctx, store, judgePath)
	testcheck.NoError(t, err)
	vision := testcheck.Must(projector.OpenActiveSession(ctx, store, judgeID, projectorPath, projector.OpenOptions{CUDA: true, MemoryReserve: reserve}))(t)
	t.Cleanup(func() { testcheck.NoError(t, vision.Close()) })
	runner := testcheck.Must(modelcli.OpenRunner(ctx, roots.Store, judgePath, inference.OpenOptions{}))(t)
	t.Cleanup(func() { testcheck.NoError(t, runner.Close()) })
	bind(runner)
	type asked struct {
		runtime  judgedRuntime
		question evaluation.GeneratedAnswerCase
	}
	var questions []asked
	for _, set := range frozen {
		var suite evaluation.GeneratedAnswerSuite
		testcheck.NoError(t, jsonfile.Decode(set.suite, &suite))
		var rendered renderManifest
		testcheck.NoError(t, jsonfile.Decode(filepath.Join(set.media, renderManifestFile), &rendered))
		runtime := judgedRuntime{
			projectionRuntime: projectionRuntime{runner: runner, vision: vision, mediaFiles: &mediaFiles{root: set.media}, ffmpeg: os.Getenv("OVERGO_FFMPEG")},
			model:             rendered.Model, recipe: rendered.Recipe, store: store,
		}
		for _, question := range suite.Cases[:judgeRepeatCases] {
			// The question names its media by the reference the binding gives it, as a compiled suite does.
			kind, name, _, _ := evaluation.ParseMediaPrompt(question.Prompt)
			reference := testcheck.Must(testcheck.Must(runtime.ResolveMedia(ctx, kind, name))(t).Reference())(t)
			question.Prompt = strings.Replace(question.Prompt, "name="+name+">", "name="+reference+">", 1)
			questions = append(questions, asked{runtime, question})
		}
	}
	greedy := testcheck.Must(sampling.New(sampling.Config{Temperature: 0}))(t)
	// answer is the judge's letter for a question and its probability of the right one: the softmax of its first
	// token's logits over the letters the question offers.
	answer := func(question asked) (string, float64) {
		var letters []string
		for line := range strings.Lines(question.question.Prompt) {
			if letter, _, found := strings.Cut(line, ". "); found && len(letter) == 1 && strings.Contains("ABCD", letter) {
				letters = append(letters, letter)
			}
		}
		tokens := testcheck.Must(question.runtime.ChoiceTokens(letters))(t)
		var first []float32
		_, text, err := question.runtime.Generate(ctx, question.question.Prompt, inference.GenerateOptions{
			MaxNewTokens: question.question.MaxTokens, Sampler: greedy,
			OnToken: func(event inference.TokenEvent) error {
				if first == nil {
					first = slices.Clone(event.Logits)
				}
				return nil
			},
		})
		testcheck.NoError(t, err)
		var total, right float64
		for index, token := range tokens {
			weight := math.Exp(float64(first[token]))
			total += weight
			if letters[index] == question.question.Answers[0] {
				right = weight
			}
		}
		return strings.TrimSpace(text), right / total
	}
	var runs [2][]float64
	var letters [2][]string
	for run := range runs {
		for _, question := range questions {
			letter, probability := answer(question)
			letters[run], runs[run] = append(letters[run], letter), append(runs[run], probability)
		}
	}
	var worst float64
	for index, question := range questions {
		worst = max(worst, math.Abs(runs[0][index]-runs[1][index]))
		if letters[0][index] != letters[1][index] {
			t.Errorf("%s: the judge answered %q and then %q", question.question.Name, letters[0][index], letters[1][index])
		}
	}
	share := func(probabilities []float64) float64 {
		var logs float64
		for _, probability := range probabilities {
			logs += math.Log(probability)
		}
		return math.Exp(logs / float64(len(probabilities)))
	}
	t.Logf("%d questions twice: largest move of a right letter's probability %.3g, soft share %.6f then %.6f", len(questions), worst, share(runs[0]), share(runs[1]))
	if worst > judgeRepeatTolerance || math.Abs(share(runs[0])-share(runs[1])) > judgeRepeatTolerance {
		t.Fatalf("the judge's probabilities moved by %g and its soft share by %g between two runs, beyond %g", worst, math.Abs(share(runs[0])-share(runs[1])), judgeRepeatTolerance)
	}
}
