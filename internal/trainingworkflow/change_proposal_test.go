package trainingworkflow

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestLinearChangeCodeFindsTheDirectionGainRisesIn holds the proposer's code
// to a corpus with a known answer: eight changes that differ along three
// directions, their gains set by the first alone. The code keeps few
// directions, predicts a run left out nearly exactly, and its direction,
// taken as a weighted sum of the centered changes, points along the first
// direction and no other. With gains that follow no direction it predicts
// nothing.
func TestLinearChangeCodeFindsTheDirectionGainRisesIn(t *testing.T) {
	t.Parallel()
	const runs, width = 8, 3
	rng := rand.New(rand.NewPCG(3, 3))
	changes := make([][]float64, runs)
	gains, noise := make([]float64, runs), make([]float64, runs)
	for run := range changes {
		// The directions differ in how far the runs spread along them: 1, 3 and 0.3.
		changes[run] = []float64{2 + rng.NormFloat64(), 3 * rng.NormFloat64(), 0.3 * rng.NormFloat64()}
		gains[run], noise[run] = 0.5*changes[run][0], rng.NormFloat64()
	}
	gram := make([]float64, runs*runs)
	mean := make([]float64, width)
	for row := range runs {
		for at, value := range changes[row] {
			mean[at] += value / runs
		}
		for column := range runs {
			for at := range width {
				gram[row*runs+column] += changes[row][at] * changes[column][at]
			}
		}
	}
	code, err := linearChangeCode(gram, gains)
	if err != nil {
		t.Fatal(err)
	}
	direction := make([]float64, width)
	for run, weight := range code.direction {
		for at := range width {
			direction[at] += weight * (changes[run][at] - mean[at])
		}
	}
	t.Logf("rank %d explains %.4f; direction %v of length %.4f; runs sit %.4f from the mean", code.rank, code.explained, direction, code.length, code.spread)
	const tolerance = 1e-9
	if code.rank > width || code.explained < 1-tolerance || direction[0] <= 0 || math.Abs(direction[1]) > tolerance*direction[0] || math.Abs(direction[2]) > tolerance*direction[0] {
		t.Fatalf("the code of rank %d explains %v and points along %v, want all of the gain and the first direction alone", code.rank, code.explained, direction)
	}
	if length := math.Hypot(direction[0], math.Hypot(direction[1], direction[2])); math.Abs(length-code.length) > tolerance*length {
		t.Fatalf("the direction is %v long and the code says %v", length, code.length)
	}
	unrelated, err := linearChangeCode(gram, noise)
	if err != nil || unrelated.explained > 0.5 {
		t.Fatalf("gains that follow no direction are explained to %v (%v)", unrelated.explained, err)
	}
	// The proposal and the random control sit as far from the mean as a typical run.
	for _, candidate := range changeCandidates(gram, gains, code, 1, 2) {
		var sum float64
		centered := make([]float64, runs)
		for run, weight := range candidate.Weights {
			sum += weight
			centered[run] = weight - 1.0/runs
		}
		if math.Abs(sum-1) > tolerance {
			t.Fatalf("%s's weights sum to %v, want a change of the base and not a rescaling of it", candidate.Name, sum)
		}
		if moved := candidate.Name == changeProposed || candidate.Name == changeRandom; moved && math.Abs(changeLength(gram, centered)-code.spread) > tolerance*code.spread {
			t.Fatalf("%s sits %v from the mean change, a typical run %v", candidate.Name, changeLength(gram, centered), code.spread)
		}
	}
	if _, err := linearChangeCode(gram, gains[:runs-1]); err == nil {
		t.Fatal("a code was fitted with a gain missing")
	}
}

// TestProposedChangeIsJudgedAgainstItsControls collects a corpus of a tiny
// model's fine-tunes and has a change proposed from it. The proposal and each
// control is a weighted sum of the corpus's changes added to the base: the
// written models hold exactly those weights, the best run is its own
// checkpoint and reads what its verdict read, and the proposal and the random
// control sit the same distance from the mean. Every candidate is scored on
// the corpus's held-out records; the control to beat is the one that scored
// best; and the proposal is kept only when the redrawn records hold its
// score above that control's. The judgment is the same judged twice, and a
// sequence length that reads other held-out inputs is refused.
func TestProposedChangeIsJudgedAgainstItsControls(t *testing.T) {
	var documents []string
	for index, pattern := range []string{"abcd", "dcba", "acbd", "bdac", "cadb", "dbca", "abdc", "cdab", "badc", "dacb", "cbad", "adcb", "bcda", "cabd", "dbac", "acdb", "abcd", "dcba", "acbd", "bdac"} {
		documents = append(documents, strings.Repeat(pattern, 8)[:8+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	const (
		window = 8
		steps  = 3
		count  = 6
	)
	runsDirectory := filepath.Join(text.root, "corpus")
	corpus, err := CollectChangeCorpus(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: runsDirectory, Steps: steps, MaximumSequence: window, Host: true,
	}, 1, count)
	if err != nil {
		t.Fatal(err)
	}
	request := ChangeProposalRequest{
		Corpus: corpus.ID, ModelDirectory: text.model, RunsDirectory: runsDirectory, OutputDirectory: filepath.Join(text.root, "candidates"), MaximumSequence: window,
	}
	judgment, err := ProposeChange(ctx, text.store, request)
	if err != nil {
		t.Fatal(err)
	}
	committed, err := changeJudgmentCodec.Require(ctx, text.store, judgment.ID)
	if err != nil || committed.Corpus != corpus.ID || len(committed.Candidates) != len(judgment.Candidates) {
		t.Fatalf("committed judgment = (%d candidates of corpus %s, %v)", len(committed.Candidates), committed.Corpus, err)
	}
	t.Logf("rank %d explains %.3f; gain spread %.4f against noise %.4f; base %.4f", judgment.Rank, judgment.Explained, judgment.GainSpread, judgment.GainNoise, judgment.Base)

	directories := make([]string, count)
	for index, change := range corpus.Changes {
		directories[index] = filepath.Join(runsDirectory, fmt.Sprintf("order-%04d", change.Order))
	}
	tensors, err := openChangeTensors(text.model, directories)
	if err != nil {
		t.Fatal(err)
	}
	defer tensors.Close()
	gram, err := changeGram(ctx, tensors)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]ChangeCandidate{}
	var distances []float64
	best := 0.0
	for _, candidate := range judgment.Candidates {
		byName[candidate.Name] = candidate
		t.Logf("%-17s %.4f weights %.3f", candidate.Name, candidate.Score, candidate.Weights)
		centered := make([]float64, count)
		var sum float64
		for run, weight := range candidate.Weights {
			sum += weight
			centered[run] = weight - 1.0/count
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("%s's weights sum to %v", candidate.Name, sum)
		}
		if candidate.Name == changeProposed || candidate.Name == changeRandom {
			distances = append(distances, changeLength(gram, centered))
		}
		if candidate.Name != changeProposed {
			best = max(best, candidate.Score)
		}
		report, found, err := artifact.ReadContent(ctx, text.store, candidate.Report)
		if err != nil || !found || len(report.Data) == 0 {
			t.Fatalf("%s's report %s is committed %t (%v)", candidate.Name, candidate.Report, found, err)
		}
		if candidate.Name == changeBestRun {
			continue
		}
		// The written model holds the base plus the weighted changes, tensor by tensor.
		written, err := safetensors.OpenSource(filepath.Join(request.OutputDirectory, candidate.Name))
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range tensors.names {
			base, changes, err := tensors.change(name)
			if err != nil {
				t.Fatal(err)
			}
			held, err := safetensors.ReadF32(written.Tensors[name])
			if err != nil {
				t.Fatal(err)
			}
			for at, value := range base {
				want := float64(value)
				for run, change := range changes {
					want += candidate.Weights[run] * float64(change[at])
				}
				if held[at] != float32(want) {
					t.Fatalf("%s holds %v at %s[%d], want the base plus its weighted changes %v", candidate.Name, held[at], name, at, float32(want))
				}
			}
		}
		if err := written.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := loadTrainableTokenizer(filepath.Join(request.OutputDirectory, candidate.Name)); err != nil {
			t.Fatalf("%s does not load as its base does: %v", candidate.Name, err)
		}
	}
	for _, name := range []string{changeProposed, changeBestRun, changeMean, changeRandom} {
		if _, found := byName[name]; !found {
			t.Fatalf("the judgment scores no %s", name)
		}
	}
	if len(distances) != 2 || math.Abs(distances[0]-distances[1]) > 1e-6*distances[0] || !(distances[0] > 0) {
		t.Fatalf("the proposal and the random control sit %v from the mean change, want one distance", distances)
	}
	// The best run is one run: its own checkpoint, read as its verdict read it.
	winner := slices.Index(byName[changeBestRun].Weights, 1)
	gains := make([]float64, count)
	for index, change := range corpus.Changes {
		gains[index] = change.Changed - change.Base
	}
	if winner != slices.Index(gains, slices.Max(gains)) || byName[changeBestRun].Score != corpus.Changes[winner].Changed || judgment.Base != corpus.Changes[0].Base {
		t.Fatalf("the best run is run %d reading %v, the corpus's best is %d reading %v", winner, byName[changeBestRun].Score, slices.Index(gains, slices.Max(gains)), corpus.Changes[winner].Changed)
	}
	if byName[judgment.Control].Score != best || judgment.Control == changeProposed ||
		math.Abs(judgment.Difference-(byName[changeProposed].Score-best)) > 1e-12 ||
		judgment.Lower > judgment.Upper || judgment.Kept != (judgment.Lower > 0) {
		t.Fatalf("judged against %s: difference %v in [%v, %v], kept %t; the best control read %v", judgment.Control, judgment.Difference, judgment.Lower, judgment.Upper, judgment.Kept, best)
	}

	again := request
	again.OutputDirectory = filepath.Join(text.root, "candidates-again")
	repeated, err := ProposeChange(ctx, text.store, again)
	if err != nil || repeated.ID != judgment.ID {
		t.Fatalf("judged again = (%s, %v), want the same judgment %s", repeated.ID, err, judgment.ID)
	}
	other := request
	other.OutputDirectory, other.MaximumSequence = filepath.Join(text.root, "candidates-other"), window-1
	if _, err := ProposeChange(ctx, text.store, other); err == nil || !strings.Contains(err.Error(), "other held-out inputs") {
		t.Fatalf("a proposal judged at another sequence length = %v, want it refused", err)
	}
}

// TestResampledDifferencesBoundAProposalAgainstItsControl holds the paired
// resampling to two cases with known answers: a proposal better on every
// held-out record is above its control in every draw, and one equal to its
// control is above it in none. The draws are the plan's, whole records.
func TestResampledDifferencesBoundAProposalAgainstItsControl(t *testing.T) {
	t.Parallel()
	corpus := ChangeCorpus{
		ID:     testutil.ArtifactID(t, artifact.KindEvidence, "corpus"),
		Inputs: testutil.ArtifactID(t, artifact.KindProfile, "held-out inputs"),
	}
	var control, better []evaluation.RecordScore
	for record := range 12 {
		name := fmt.Sprintf("record-%02d", record)
		control = append(control, evaluation.RecordScore{Record: name, Value: 0.5 + 0.01*float64(record)})
		better = append(better, evaluation.RecordScore{Record: name, Value: 0.52 + 0.012*float64(record)})
	}
	plan, differences, err := resampledDifferences(t.Context(), corpus, better, control, 1, 7, 9)
	if err != nil || len(differences) != changeResamplingDraws || len(plan.Units) != len(control) || !slices.IsSorted(differences) {
		t.Fatalf("resampled %d differences over %d units (%v)", len(differences), len(plan.Units), err)
	}
	if differences[0] <= 0 || differences[0] == differences[len(differences)-1] {
		t.Fatalf("a proposal better on every record reads from %v to %v over the draws, want it above its control in every draw and the draws to differ", differences[0], differences[len(differences)-1])
	}
	_, lower, err := resampledDifferences(t.Context(), corpus, better, control, -1, 7, 9)
	if err != nil || lower[len(lower)-1] >= 0 {
		t.Fatalf("under a metric that is better smaller, the larger score reads %v at best (%v), want it below its control", lower[len(lower)-1], err)
	}
	_, equal, err := resampledDifferences(t.Context(), corpus, control, control, 1, 7, 9)
	if err != nil || equal[0] != 0 || equal[len(equal)-1] != 0 {
		t.Fatalf("a proposal equal to its control reads %v to %v (%v), want no difference in any draw", equal[0], equal[len(equal)-1], err)
	}
	if _, _, err := resampledDifferences(t.Context(), corpus, better, control[1:], 1, 7, 9); err == nil {
		t.Fatal("candidates scored on different records were compared")
	}
}
