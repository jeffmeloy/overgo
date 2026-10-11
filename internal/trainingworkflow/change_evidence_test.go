package trainingworkflow

import (
	"cmp"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestEvidenceGainsAreFittedBesideThePoolsGains holds what a proposer shown
// past verdicts fits its code to, on four runs with gains written by hand.
// Three of the runs are judged by a shown verdict. Each kind of gain is taken
// about its mean in units of its spread over the runs it covers, and the two
// are added; the run no shown verdict judged keeps its pool gain alone. A
// metric that is better when smaller turns its verdicts' gains over. One
// judged run, or judged runs whose verdicts recorded one gain, leave nothing
// to fit, and the pool's gains are returned as they were.
func TestEvidenceGainsAreFittedBesideThePoolsGains(t *testing.T) {
	t.Parallel()
	verdict := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvaluation, name) }
	corpus := ChangeCorpus{Direction: runrecord.DirectionMaximize, Changes: []WeightChange{
		{Order: 1, Verdict: verdict("one"), Base: 0.5, Changed: 0.6},
		{Order: 2, Verdict: verdict("two"), Base: 0.5, Changed: 0.5},
		{Order: 3, Verdict: verdict("three"), Base: 0.5, Changed: 0.7},
		{Order: 4, Verdict: verdict("four"), Base: 0.5, Changed: 0.9},
	}}
	gains := []float64{1, 2, 3, 6}
	shown := []artifact.ID{corpus.Changes[0].Verdict, corpus.Changes[1].Verdict, corpus.Changes[2].Verdict, verdict("of another corpus")}
	// The pool's gains stand -2, -1, 0 and 3 from their mean of 3, in units of
	// the root of 14 over 3; the three verdicts' gains of 0.1, 0 and 0.2 stand
	// 0, -1 and 1 from their mean in units of their spread, 0.1.
	spread := math.Sqrt(14.0 / 3)
	near := func(got, want []float64) bool {
		for index := range want {
			if len(got) != len(want) || math.Abs(got[index]-want[index]) > 1e-9 {
				return false
			}
		}
		return true
	}
	fitted, used := conditionedGains(corpus, gains, shown)
	if want := []float64{-2 / spread, -1/spread - 1, 1, 3 / spread}; used != 3 || !near(fitted, want) {
		t.Fatalf("three shown verdicts fit %v for %d runs, want %v for 3", fitted, used, want)
	}
	corpus.Direction = runrecord.DirectionMinimize
	fitted, used = conditionedGains(corpus, gains, shown)
	if want := []float64{-2 / spread, -1/spread + 1, -1, 3 / spread}; used != 3 || !near(fitted, want) {
		t.Fatalf("under a metric that is better smaller the verdicts fit %v for %d runs, want %v for 3", fitted, used, want)
	}
	for name, few := range map[string][]artifact.ID{
		"none":                  nil,
		"one judged run":        shown[:1],
		"another corpus's only": shown[3:],
	} {
		if fitted, used := conditionedGains(corpus, gains, few); used != 0 || !slices.Equal(fitted, gains) {
			t.Fatalf("%s: fitted %v for %d runs, want the pool's gains as they were", name, fitted, used)
		}
	}
	corpus.Changes[1].Changed, corpus.Changes[2].Changed = corpus.Changes[0].Changed, corpus.Changes[0].Changed
	if fitted, used := conditionedGains(corpus, gains, shown); used != 0 || !slices.Equal(fitted, gains) {
		t.Fatalf("verdicts that recorded one gain fitted %v for %d runs, want the pool's gains as they were", fitted, used)
	}
}

// TestEvidenceConditionedProposalIsCompared runs the three arms a proposer's
// past evidence is judged by, on a corpus that grew: six runs, and two more
// after a moment the journal is read back to. One arm is shown nothing. One is
// shown the held-out verdicts the store holds as it stands, which judge all
// eight runs. One is shown the store as it stood at that moment, which judged
// six, and the two runs tried after it are fitted by what they read on the
// pool alone.
//
// Each arm's judgment records the verdicts it was shown, by their identities,
// and how many of its runs they judged. An arm shown verdicts scores, beside
// its proposal, the proposal the same code makes of the pool's gains alone,
// which is the first arm's proposal weight for weight and reading for
// reading, and says how far apart the two stand on the pool's selection part
// with the bounds of that over the redrawn documents. Its own proposal is the
// one the code fitted to the gains with the evidence beside them makes. A
// proposer cannot be shown something the store does not hold as a held-out
// verdict.
func TestEvidenceConditionedProposalIsCompared(t *testing.T) {
	ctx := t.Context()
	text, first, _, request := pooledCorpus(t)
	_, between := text.store.Head()
	const grown = pooledRuns + 2
	corpus, err := CollectChangeCorpus(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: request.RunsDirectory, Steps: pooledSteps, MaximumSequence: pooledWindow, Host: true,
	}, 1, grown)
	if err != nil {
		t.Fatal(err)
	}
	request.Corpus = corpus.ID
	sorted := func(named []artifact.ID) []artifact.ID {
		named = slices.Clone(named)
		slices.SortFunc(named, func(left, right artifact.ID) int { return cmp.Compare(left.String(), right.String()) })
		return named
	}
	read := func(through uint64, stood ChangeCorpus) ([]artifact.ID, uint64) {
		t.Helper()
		shown, bytes, err := evaluation.PastHeldoutVerdicts(ctx, text.store, through, 0)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]artifact.ID, len(stood.Changes))
		for index, change := range stood.Changes {
			want[index] = change.Verdict
		}
		if !slices.Equal(sorted(shown), sorted(want)) {
			t.Fatalf("through sequence %d the store names %d verdicts, want the %d of the corpus as it stood", through, len(shown), len(want))
		}
		return shown, bytes
	}
	now, nowBytes := read(0, corpus)
	then, thenBytes := read(between, first)
	if len(then) != pooledRuns || len(now) != grown || thenBytes >= nowBytes {
		t.Fatalf("the store names %d verdicts in %d bytes as it stood and %d in %d as it stands", len(then), thenBytes, len(now), nowBytes)
	}

	arm := func(name string, shown []artifact.ID) (ChangeJudgment, map[string]ChangeCandidate) {
		t.Helper()
		asked := request
		asked.Shown, asked.OutputDirectory = shown, filepath.Join(text.root, "arm-"+name)
		judgment, err := ProposeChange(ctx, text.store, asked)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if committed, err := changeJudgmentCodec.Require(ctx, text.store, judgment.ID); err != nil || !slices.Equal(committed.Shown, shown) || committed.Used != judgment.Used {
			t.Fatalf("%s: the committed judgment was shown %d verdicts and used %d (%v), want %d and %d", name, len(committed.Shown), committed.Used, err, len(shown), judgment.Used)
		}
		candidates := map[string]ChangeCandidate{}
		for _, candidate := range judgment.Candidates {
			candidates[candidate.Name] = candidate
		}
		t.Logf("%-8s shown %d verdicts, %d of its runs'; proposed %.4f, without evidence %.4f; apart %.4f in [%.4f, %.4f]", name, len(shown), judgment.Used,
			candidates[changeProposed].Score, candidates[changeUnconditioned].Score, judgment.EvidenceDifference, judgment.EvidenceLower, judgment.EvidenceUpper)
		return judgment, candidates
	}
	none, plain := arm("none", nil)
	if _, scored := plain[changeUnconditioned]; scored || none.Used != 0 || none.EvidenceDifference != 0 || none.EvidenceLower != 0 || none.EvidenceUpper != 0 {
		t.Fatalf("the arm shown nothing records %+v", none)
	}

	// What each arm's code is fitted to, taken apart from the proposer: the
	// runs' gains on the pool's development part, their changes' inner
	// products, and the evidence beside the gains.
	pool, err := evaluation.RequireHeldoutPool(ctx, text.store, request.Acceptance)
	if err != nil {
		t.Fatal(err)
	}
	base, examples, err := poolExamples(ctx, text.store, pool.Acceptance, text.model)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := documentLikelihoods(base, examples, pool.Development)
	if err != nil {
		t.Fatal(err)
	}
	directories, gains := make([]string, grown), make([]float64, grown)
	for index, change := range corpus.Changes {
		directories[index] = runDirectory(request.RunsDirectory, change.Order)
		model, err := loadTrainableModel(directories[index])
		if err != nil {
			t.Fatal(err)
		}
		scores, err := documentLikelihoods(model, examples, pool.Development)
		if err != nil {
			t.Fatal(err)
		}
		gains[index] = scoreMean(scores) - scoreMean(baseline)
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
	seedHigh, seedLow := corpusSeeds(corpus.ID)
	var proposals [][]float64
	for _, shown := range []struct {
		name     string
		verdicts []artifact.ID
		judged   int
	}{{"current", now, grown}, {"cutoff", then, pooledRuns}} {
		judgment, candidates := arm(shown.name, shown.verdicts)
		if judgment.Used != shown.judged {
			t.Fatalf("%s: the verdicts shown judged %d of the corpus's runs, want %d", shown.name, judgment.Used, shown.judged)
		}
		without, proposed := candidates[changeUnconditioned], candidates[changeProposed]
		if !slices.Equal(without.Weights, plain[changeProposed].Weights) || without.Report != plain[changeProposed].Report || without.Score != plain[changeProposed].Score {
			t.Fatalf("%s: the proposal without evidence is %v scoring %v, and the arm shown nothing proposed %v scoring %v", shown.name, without.Weights, without.Score, plain[changeProposed].Weights, plain[changeProposed].Score)
		}
		fitted, used := conditionedGains(corpus, gains, shown.verdicts)
		code, err := linearChangeCode(gram, fitted)
		if err != nil || used != shown.judged {
			t.Fatalf("%s: the evidence fits %d runs (%v)", shown.name, used, err)
		}
		if want := changeCandidates(gram, gains, code, seedHigh, seedLow)[0].Weights; !slices.Equal(proposed.Weights, want) || slices.Equal(proposed.Weights, without.Weights) {
			t.Fatalf("%s: proposed %v, want the proposal of the code fitted with the evidence, %v, and not the one without it", shown.name, proposed.Weights, want)
		}
		if judgment.EvidenceDifference != proposed.Score-without.Score || judgment.EvidenceLower > judgment.EvidenceUpper {
			t.Fatalf("%s: the proposal stands %v from the one without evidence in [%v, %v]; they scored %v and %v", shown.name, judgment.EvidenceDifference, judgment.EvidenceLower, judgment.EvidenceUpper, proposed.Score, without.Score)
		}
		proposals = append(proposals, proposed.Weights)
	}
	if slices.Equal(proposals[0], proposals[1]) {
		t.Fatal("the arm shown the store as it stood proposed what the arm shown it as it stands proposed, with two runs' verdicts fewer")
	}

	absent := request
	absent.Shown, absent.OutputDirectory = append(slices.Clone(now), corpus.ID), filepath.Join(text.root, "arm-absent")
	if judgment, err := ProposeChange(ctx, text.store, absent); err == nil {
		t.Fatalf("a proposer shown %s, which is no held-out verdict, was judged: %s", corpus.ID, judgment.ID)
	}
}
