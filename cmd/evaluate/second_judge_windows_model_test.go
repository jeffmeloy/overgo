//go:build windows && modeltest

package main

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	// secondJudgeOutputs is how many pictures and how many clips both judges read: the first of each calibration suite.
	secondJudgeOutputs = 32
	// agreementZ is the standard normal's 95th percentile: the floor, declared before the run, is the Kendall's tau
	// two judges ranking independently exceed one time in twenty, with n outputs, z times its deviation
	// sqrt(2(2n+5)/(9n(n-1))): 0.204 at 32.
	agreementZ = 1.6448536269514722
	// agreementResamples is how many times the outputs are drawn again, with replacement, for tau's interval.
	agreementResamples = 2000
	// agreementInterval is the share of the resampled taus the printed interval holds.
	agreementInterval = 0.95
	// agreementSeed seeds the resampling, so the interval printed is the same each run.
	agreementSeed = 20261109
)

// TestASecondJudgeRanksTheCasesAlike measures whether a judge of another family ranks the media as Qwen3.8-27B does.
// Both judges answer the questions the calibration suites ask of their first 32 pictures and 32 clips, each question
// scored by its choices, the lettered options its prompt offers, so each answer carries the judge's probability of the
// right one. An output's score is the mean of those probabilities over its questions, and the two judges' scores of a
// medium's outputs agree in rank by Kendall's tau-b above the floor declared before the run: the tau independent
// rankings exceed one time in twenty (agreementZ). The tau's interval over agreementResamples draws of the outputs is
// printed, and every question one judge answers right and the other wrong is listed.
func TestASecondJudgeRanksTheCasesAlike(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	root := t.TempDir()
	pictureSuite, clipSuite := calibrationSuites(roots)
	suites := []evaluation.GeneratedAnswerSuite{
		withLetteredChoices(degradedSuite(t, ctx, ffmpeg, pictureSuite, root, secondJudgeOutputs, nil, nil)),
		withLetteredChoices(degradedSuite(t, ctx, ffmpeg, clipSuite, root, secondJudgeOutputs, nil, nil)),
	}
	qwen := filepath.Join(roots.ResolveModelPath("Qwen3.8-27B-UD-Q6_K"), "Qwen3.8-27B-UD-Q6_K.gguf")
	gemma := roots.ResolveModelPath("gemma-4-12B-it")
	judges := []struct{ name, path, projector string }{
		{"Qwen3.8-27B", qwen, filepath.Join(filepath.Dir(qwen), "mmproj-BF16.gguf")},
		{"gemma-4-12B-it", gemma, gemma},
	}
	// read[judge][suite] is each case's answer; a judge is closed before the next opens, since the two do not fit the
	// card together.
	read := make([][]map[string]judgedAnswer, len(judges))
	for at, judge := range judges {
		t.Run(judge.name, func(t *testing.T) {
			opened := openJudge(t, ctx, roots, root, judge.path, judge.projector)
			for _, suite := range suites {
				read[at] = append(read[at], opened.judged(t, ctx, suite))
			}
		})
	}
	if t.Failed() {
		return
	}
	for index, suite := range suites {
		outputs, cases := outputCases(suite)
		scores := make([][]float64, len(judges))
		for at := range judges {
			for _, output := range outputs {
				var total float64
				for _, name := range cases[output] {
					total += read[at][index][name].probability
				}
				scores[at] = append(scores[at], total/float64(len(cases[output])))
			}
		}
		n := float64(len(outputs))
		floor := agreementZ * math.Sqrt(2*(2*n+5)/(9*n*(n-1)))
		tau := kendallTauB(scores[0], scores[1])
		low, high := resampledTau(scores[0], scores[1])
		t.Logf("%s: %d outputs, Kendall's tau-b %.3f (%.0f%% of %d resamples %.3f to %.3f), floor %.3f",
			suite.Source, len(outputs), tau, agreementInterval*100, agreementResamples, low, high, floor)
		disagreed := 0
		for _, output := range outputs {
			for _, name := range cases[output] {
				first, second := read[0][index][name], read[1][index][name]
				if first.right != second.right {
					disagreed++
					t.Logf("  %s: %s right %t (%.2f), %s right %t (%.2f)", name, judges[0].name, first.right, first.probability,
						judges[1].name, second.right, second.probability)
				}
			}
		}
		t.Logf("%s: the judges disagree on %d of %d questions", suite.Source, disagreed, len(suite.Cases))
		if math.IsNaN(tau) || tau <= floor {
			t.Errorf("%s and %s rank the %d outputs of %s with Kendall's tau-b %.3f, at or below the floor %.3f",
				judges[0].name, judges[1].name, len(outputs), suite.Source, tau, floor)
		}
	}
}

// judgedAnswer is one case's answer: whether the judge chose the right choice, and its probability of it.
type judgedAnswer struct {
	right       bool
	probability float64
}

// judged has the judge answer every case of suite and reads, by case, its choice and its probability of the right one
// from the report the evaluation kept.
func (judge calibration) judged(t *testing.T, ctx context.Context, suite evaluation.GeneratedAnswerSuite) map[string]judgedAnswer {
	t.Helper()
	compiled := testcheck.Must(judge.session.campaign.CompileSuite(ctx, testcheck.Must(json.Marshal(suite))(t)))(t)
	result := testcheck.Must(judge.session.campaign.Evaluate(ctx, compiled))(t)
	evaluated := testcheck.Must(runrecord.RequireEvaluation(ctx, judge.answers, result.Evaluation))(t)
	bound := testcheck.Must(runrecord.RequireRun(ctx, judge.answers, evaluated.Run))(t)
	if len(bound.Outputs) != 1 {
		t.Fatalf("the evaluation of %s wrote %d reports, want one", suite.Source, len(bound.Outputs))
	}
	content, found, err := artifact.ReadContent(ctx, judge.answers, bound.Outputs[0])
	if err != nil || !found {
		t.Fatalf("the report of %s cannot be read: %v", suite.Source, err)
	}
	var report struct {
		Observations []struct {
			Name        string   `json:"name"`
			Accepted    bool     `json:"accepted"`
			Probability *float64 `json:"probability"`
		} `json:"observations"`
	}
	testcheck.NoError(t, json.Unmarshal(content.Data, &report))
	answers := make(map[string]judgedAnswer, len(report.Observations))
	for _, observation := range report.Observations {
		if observation.Probability == nil {
			t.Fatalf("%s's answer to %s carries no probability", suite.Source, observation.Name)
		}
		answers[observation.Name] = judgedAnswer{right: observation.Accepted, probability: *observation.Probability}
	}
	if len(answers) != len(suite.Cases) {
		t.Fatalf("the report of %s answers %d of its %d cases", suite.Source, len(answers), len(suite.Cases))
	}
	return answers
}

// withLetteredChoices gives each case that has no choices the letters of the options its prompt offers, one a line
// as "A. yes", so it is scored by its choices and its answer carries the judge's probability of the right one.
func withLetteredChoices(suite evaluation.GeneratedAnswerSuite) evaluation.GeneratedAnswerSuite {
	cases := slices.Clone(suite.Cases)
	for index, testCase := range cases {
		if len(testCase.Choices) > 0 {
			continue
		}
		for line := range strings.Lines(testCase.Prompt) {
			letter, option, offered := strings.Cut(line, ". ")
			if offered && len(letter) == 1 && letter >= "A" && letter <= "Z" && strings.TrimSpace(option) != "" {
				cases[index].Choices = append(cases[index].Choices, letter)
			}
		}
	}
	suite.Cases = cases
	return suite
}

// outputCases are the outputs a suite asks about, in its order, and the names of each one's cases.
func outputCases(suite evaluation.GeneratedAnswerSuite) ([]string, map[string][]string) {
	var outputs []string
	cases := map[string][]string{}
	for _, testCase := range suite.Cases {
		if _, seen := cases[testCase.Group]; !seen {
			outputs = append(outputs, testCase.Group)
		}
		cases[testCase.Group] = append(cases[testCase.Group], testCase.Name)
	}
	return outputs, cases
}

// kendallTauB is Kendall's tau-b of two scorings of the same outputs: the concordant pairs less the discordant, over
// the geometric mean of the pairs each scoring does not tie. NaN where either ties every pair.
func kendallTauB(first, second []float64) float64 {
	var concordant, discordant, tiedFirst, tiedSecond, pairs float64
	for i := range first {
		for j := i + 1; j < len(first); j++ {
			pairs++
			across, down := cmpSign(first[i], first[j]), cmpSign(second[i], second[j])
			switch {
			case across == 0 || down == 0:
				if across == 0 {
					tiedFirst++
				}
				if down == 0 {
					tiedSecond++
				}
			case across == down:
				concordant++
			default:
				discordant++
			}
		}
	}
	return (concordant - discordant) / math.Sqrt((pairs-tiedFirst)*(pairs-tiedSecond))
}

// cmpSign is -1, 0 or 1 as a is below, at or above b.
func cmpSign(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// resampledTau is the interval holding agreementInterval of Kendall's tau-b over agreementResamples draws of the
// outputs with replacement, a draw whose tau is undefined left out.
func resampledTau(first, second []float64) (low, high float64) {
	random := rand.New(rand.NewPCG(agreementSeed, agreementSeed))
	var taus []float64
	drawnFirst, drawnSecond := make([]float64, len(first)), make([]float64, len(second))
	for range agreementResamples {
		for index := range drawnFirst {
			at := random.IntN(len(first))
			drawnFirst[index], drawnSecond[index] = first[at], second[at]
		}
		if tau := kendallTauB(drawnFirst, drawnSecond); !math.IsNaN(tau) {
			taus = append(taus, tau)
		}
	}
	slices.Sort(taus)
	tail := (1 - agreementInterval) / 2
	return taus[int(tail*float64(len(taus)))], taus[min(len(taus)-1, int((1-tail)*float64(len(taus))))]
}
