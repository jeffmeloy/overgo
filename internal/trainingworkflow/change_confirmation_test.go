package trainingworkflow

import (
	"cmp"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// What the pooled corpus fixture is made of: the documents of its objective,
// the held-out ones its verdicts score, the runs of its corpus, their updates
// and their sequence length, and the gain its pool is sized to show, in
// standard deviations of the paired per-document difference.
const (
	pooledDocuments = 96
	pooledScored    = 8
	pooledRuns      = 6
	pooledSteps     = 3
	pooledWindow    = 8
	pooledGain      = 1
)

// pooledCorpus registers a text objective whose verdicts score a few of its
// held-out documents, collects a corpus of a tiny model's fine-tunes under
// it, and publishes the pool of the held-out documents no verdict read. It
// returns the objective's fixture, the corpus, the pool and the request that
// has a change proposed from the corpus on the pool.
func pooledCorpus(t *testing.T) (registeredText, ChangeCorpus, evaluation.HeldoutPool, ChangeProposalRequest) {
	t.Helper()
	ctx := t.Context()
	patterns := []string{"abcd", "dcba", "acbd", "bdac", "cadb", "dbca", "abdc", "cdab", "badc", "dacb", "cbad", "adcb", "bcda", "cabd", "dbac", "acdb"}
	var documents []string
	for index := range pooledDocuments {
		// No two documents are the same text: each is longer than the one before.
		documents = append(documents, strings.Repeat(patterns[index%len(patterns)], pooledDocuments)[:pooledWindow+1+index])
	}
	text := registerScoredTextObjective(t, documents, trainingprogram.MetricTokenAccuracy, pooledScored)
	runs := filepath.Join(text.root, "corpus")
	corpus, err := CollectChangeCorpus(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: runs, Steps: pooledSteps, MaximumSequence: pooledWindow, Host: true,
	}, 1, pooledRuns)
	if err != nil {
		t.Fatal(err)
	}
	objective, err := trainingprogram.LoadObjective(ctx, text.store, corpus.Objective)
	if err != nil {
		t.Fatal(err)
	}
	basis := make([]artifact.ID, len(corpus.Changes))
	for index, change := range corpus.Changes {
		basis[index] = change.Verdict
	}
	pool, err := evaluation.BuildHeldoutPool(ctx, text.store, evaluation.HeldoutPoolRequest{
		Objective: objective, Base: corpus.Model, Gain: pooledGain, Basis: basis, Sequence: pooledWindow, Weights: [len(evaluation.PoolParts)]uint64{2, 2, 2, 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, text.store, pool.Batch); err != nil {
		t.Fatal(err)
	}
	t.Logf("pool of %v documents in %v answers %d confirmations", pool.Documents, evaluation.PoolParts, pool.Acceptance.Confirmations)
	return text, corpus, pool, ChangeProposalRequest{
		Corpus: corpus.ID, Acceptance: pool.Acceptance.ID, ModelDirectory: text.model, RunsDirectory: runs,
		OutputDirectory: filepath.Join(text.root, "candidates"),
	}
}

// TestConfirmationIsChargedAndEntersNoChoice holds a proposal to the three
// parts of its pool and a confirmation to the pool's budget.
//
// The code is fitted on the development part, the candidates are ranked on
// the selection part, and the promotion part enters neither: with the
// promotion documents' tokens changed the judgment is the same judgment, with
// the selection documents' changed the candidates keep their weights and
// change places, and with the development documents' changed their weights
// move.
//
// A confirmation is charged before it reads: each one leaves the promotion
// part one fewer to answer, its verdict is what the promotion documents say
// of the candidate against the base, and the first past the budget is
// refused. What is left is counted from the charges the store holds, so a
// second handle on the store, as a process started again has, reads the same
// balance and the same recorded verdict and charges nothing. Neither that
// verdict nor the refusal past the budget prepares a model or a document:
// both are answered with the documents' files taken away. A candidate
// whose directory holds other weights than its judgment ranked is refused,
// and costs nothing; so is a proposal that names no pool, and a confirmation
// that names no candidate. The documents are read as far as the pool's
// acceptance fixed, by the proposal and by the confirmation alike: neither
// request names a window of its own.
func TestConfirmationIsChargedAndEntersNoChoice(t *testing.T) {
	ctx := t.Context()
	text, corpus, built, request := pooledCorpus(t)
	pool, err := evaluation.RequireHeldoutPool(ctx, text.store, built.Acceptance.ID)
	if err != nil {
		t.Fatal(err)
	}
	base, examples, err := poolExamples(ctx, text.store, pool.Acceptance, text.model)
	if err != nil {
		t.Fatal(err)
	}
	judgment, err := ProposeChange(ctx, text.store, request)
	if err != nil {
		t.Fatal(err)
	}
	membership, found, err := dataset.LoadMembership(ctx, text.store, built.Partition.Promotion)
	if err != nil || !found {
		t.Fatalf("the promotion part is committed %t (%v)", found, err)
	}
	var promotion []string
	for _, record := range membership.Records {
		promotion = append(promotion, record.ID)
	}

	// relabelled is the examples with every token of the named documents moved shift places along the vocabulary.
	relabelled := func(documents []string, shift int) map[string]HeldoutExample {
		changed := make(map[string]HeldoutExample, len(examples))
		maps.Copy(changed, examples)
		for _, document := range documents {
			example := changed[document]
			example.Input = slices.Clone(example.Input)
			for at, token := range example.Input {
				example.Input[at] = float64((int(token) + shift) % base.Dims.Vocab)
			}
			changed[document] = example
		}
		return changed
	}
	weights := func(judged ChangeJudgment) (all []float64) {
		for _, candidate := range judged.Candidates {
			all = append(all, candidate.Weights...)
		}
		return all
	}
	places := func(judged ChangeJudgment) (names []string) {
		ranked := slices.Clone(judged.Candidates)
		slices.SortStableFunc(ranked, func(left, right ChangeCandidate) int { return cmp.Compare(right.Score, left.Score) })
		for _, candidate := range ranked {
			names = append(names, candidate.Name)
		}
		return names
	}
	t.Logf("fitted on %d documents, ranked on %d: %v; %d promotion documents", len(pool.Development), len(pool.Selection), places(judgment), len(promotion))
	replaced, refitted := false, false
	for shift := 1; shift < base.Dims.Vocab; shift++ {
		// Each proposal writes its candidates in a directory of its own.
		relabel := func(part string, documents []string) (ChangeJudgment, error) {
			elsewhere := request
			elsewhere.OutputDirectory = filepath.Join(text.root, fmt.Sprintf("%s-moved-%d", part, shift))
			return proposeOnPool(ctx, text.store, elsewhere, corpus, pool, base, relabelled(documents, shift))
		}
		blind, err := relabel("promotion", promotion)
		if err != nil || blind.ID != judgment.ID {
			t.Fatalf("with the promotion documents' tokens moved %d places the proposal is judgment %s (%v), want the same judgment %s: the promotion part entered a choice", shift, blind.ID, err, judgment.ID)
		}
		ranked, err := relabel("selection", pool.Selection)
		if err != nil || !slices.Equal(weights(ranked), weights(judgment)) || ranked.ID == judgment.ID {
			t.Fatalf("with the selection documents' tokens moved %d places the candidates' weights moved or the judgment did not (%v): weights are fitted on the development part and scores read on the selection part", shift, err)
		}
		replaced = replaced || !slices.Equal(places(ranked), places(judgment))
		fitted, err := relabel("development", pool.Development)
		if err != nil {
			t.Fatal(err)
		}
		refitted = refitted || !slices.Equal(weights(fitted), weights(judgment))
	}
	if !replaced || !refitted {
		t.Fatalf("over every move of the tokens, the selection documents changed the candidates' places %t and the development documents their weights %t, want both", replaced, refitted)
	}

	// Two candidates with the same weights are one candidate to a confirmation: it names weights.
	var distinct []ChangeCandidate
	for _, candidate := range judgment.Candidates {
		if !slices.ContainsFunc(distinct, func(held ChangeCandidate) bool { return held.Report == candidate.Report }) {
			distinct = append(distinct, candidate)
		}
	}
	budget := built.Acceptance.Confirmations
	if int(budget) >= len(distinct) {
		t.Fatalf("the fixture's pool answers %d confirmations and its judgment has %d candidates of different weights; the test needs one past the budget", budget, len(distinct))
	}
	confirm := func(store *overgodb.Store, candidate string) (evaluation.PoolConfirmation, uint64, error) {
		return ConfirmChange(ctx, store, ChangeConfirmationRequest{
			Judgment: judgment.ID, Candidate: candidate, ModelDirectory: request.ModelDirectory, RunsDirectory: request.RunsDirectory,
			OutputDirectory: request.OutputDirectory,
		})
	}
	// said is what the promotion documents say of a candidate, read here apart from any confirmation.
	baseline, err := documentLikelihoods(base, examples, promotion)
	if err != nil {
		t.Fatal(err)
	}
	said := func(candidate ChangeCandidate) string {
		model, err := loadTrainableModel(candidateDirectory(corpus, request.RunsDirectory, request.OutputDirectory, candidate))
		if err != nil {
			t.Fatal(err)
		}
		read, err := documentLikelihoods(model, examples, promotion)
		if err != nil {
			t.Fatal(err)
		}
		differences := pairedDifferences(read, baseline)
		switch statistic := (scoreMean(read) - scoreMean(baseline)) / standardError(differences); {
		case statistic > built.Acceptance.Critical:
			return "gain"
		case statistic < -built.Acceptance.Critical:
			return "loss"
		}
		return "not enough evidence"
	}

	// The candidate past the budget is given another's weights first: it is refused, and nothing is charged for it.
	past := distinct[budget]
	tampered, err := IdentifyModelWeights(candidateDirectory(corpus, request.RunsDirectory, request.OutputDirectory, past))
	if err != nil {
		t.Fatal(err)
	}
	other, err := IdentifyModelWeights(candidateDirectory(corpus, request.RunsDirectory, request.OutputDirectory, distinct[0]))
	if err != nil {
		t.Fatal(err)
	}
	kept, err := os.ReadFile(tampered)
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := os.ReadFile(other)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tampered, foreign, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := confirm(text.store, past.Name); err == nil || !strings.Contains(err.Error(), "nothing was charged") {
		t.Fatalf("a candidate whose directory holds other weights than its judgment ranked = %v, want it refused", err)
	}
	if err := os.WriteFile(tampered, kept, 0o600); err != nil {
		t.Fatal(err)
	}

	var first evaluation.PoolConfirmation
	for index, candidate := range distinct[:budget] {
		confirmation, left, err := confirm(text.store, candidate.Name)
		if err != nil || left != budget-uint64(index)-1 || confirmation.Verdict != said(candidate) || confirmation.Acceptance != built.Acceptance.ID {
			t.Fatalf("confirmation %d, of %s = (%q, %d left, %v), want %q and %d left of %d", index+1, candidate.Name, confirmation.Verdict, left, err, said(candidate), budget-uint64(index)-1, budget)
		}
		t.Logf("%s: %s; the promotion part answers %d more", candidate.Name, confirmation.Verdict, left)
		if index == 0 {
			first = confirmation
		}
	}
	if _, _, err := confirm(text.store, past.Name); err == nil || !strings.Contains(err.Error(), "fresh partition") {
		t.Fatalf("the confirmation past the budget, of %s = %v, want it refused until a fresh partition is published", past.Name, err)
	}

	// With the documents' files gone nothing can be prepared, and nothing needs
	// to be: the models and the documents are prepared by a reading alone, and
	// a confirmation recorded before, or one the budget cannot pay for, is
	// answered below without one.
	if err := os.Rename(filepath.Join(text.root, "corpus"), filepath.Join(text.root, "corpus-away")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := poolExamples(ctx, text.store, pool.Acceptance, text.model); err == nil {
		t.Fatal("the pool's documents were prepared with their files gone, so their absence holds nothing")
	}

	// A second handle replays the store's log, as a process started again does.
	restarted, err := overgodb.OpenReadOnly(filepath.Join(text.root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer restarted.Close()
	recorded, left, err := confirm(restarted, distinct[0].Name)
	if err != nil || recorded.ID != first.ID || left != 0 {
		t.Fatalf("a process started again reads (%s, %d left, %v), want the recorded confirmation %s and nothing left", recorded.ID, left, err, first.ID)
	}
	if _, _, err := confirm(restarted, past.Name); err == nil || !strings.Contains(err.Error(), "fresh partition") {
		t.Fatalf("a process started again confirms past the budget: %v", err)
	}

	// Without a pool nothing is proposed, and without a candidate nothing is confirmed.
	unpooled := request
	unpooled.Acceptance = artifact.ID{}
	if _, err := ProposeChange(ctx, text.store, unpooled); err == nil || !strings.Contains(err.Error(), "held-out pool") {
		t.Fatalf("a proposal that names no pool = %v, want it refused", err)
	}
	unread := func([]string) ([]float64, error) {
		return nil, fmt.Errorf("the promotion part was read for a confirmation that names no candidate")
	}
	if _, _, err := evaluation.ConfirmOnPool(ctx, text.store, built.Acceptance.ID, artifact.ID{}, unread); err == nil || strings.Contains(err.Error(), "was read") {
		t.Fatalf("a confirmation that names no candidate = %v, want it refused before anything is read", err)
	}
}
