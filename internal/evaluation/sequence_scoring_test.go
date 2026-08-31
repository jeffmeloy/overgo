package evaluation

import (
	"context"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
)

// TestSequenceScoringPerplexity pins the held-out-corpus suite: each
// sequence scores as its full-text log-likelihood, the report
// aggregates mean NLL per token and perplexity over every scored
// token, a non-finite or positive likelihood refuses, and the report
// publishes through the campaign ledger.
func TestSequenceScoringPerplexity(t *testing.T) {
	compiled, err := CompileSequenceScoring(SequenceScoringSuite{
		Kind: SequenceScoringKind, Schema: "carbon/dna-heldout/v1", Source: "store/dna",
		Cases: []SequenceScoringCase{
			{Name: "seq-1", Group: "mrna_evo2", Text: "ACGTACGT"},
			{Name: "seq-2", Group: "mrna_evo2", Text: "TTGGCCAA"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution:   ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan, err := BindSequenceScoring(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)

	// Two sequences of 4 tokens each at total log-probability -4 and -8:
	// mean NLL per token = 12/8 = 1.5, perplexity = e^1.5.
	scorer := &sequenceFixtureScorer{scores: []sequencescore.Score{
		{LogProbability: -4, Tokens: 4}, {LogProbability: -8, Tokens: 4},
	}}
	report, err := EvaluateSequenceScoring(t.Context(), store, scorer, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.TokensScored != 8 || math.Abs(report.MeanNLLPerToken-1.5) > 1e-12 ||
		math.Abs(report.Perplexity-math.Exp(1.5)) > 1e-9 {
		t.Fatalf("report = %+v", report)
	}
	if len(report.Observations) != 2 || report.Observations[0].Group != "mrna_evo2" ||
		scorer.prompts[0] != "" || scorer.continuations[0] != "ACGTACGT" {
		t.Fatalf("observations = %+v prompts=%v", report.Observations, scorer.prompts)
	}
	if _, found, err := artifact.ReadContent(t.Context(), store, report.ID); err != nil || !found {
		t.Fatalf("published report = (%t, %v)", found, err)
	}

	// A positive log-likelihood is not a likelihood; the suite refuses.
	broken := &sequenceFixtureScorer{scores: []sequencescore.Score{{LogProbability: 1, Tokens: 4}}}
	if _, err := EvaluateSequenceScoring(t.Context(), store, broken, compiled, plan); err == nil {
		t.Fatal("positive likelihood scored")
	}
}

type sequenceFixtureScorer struct {
	scores        []sequencescore.Score
	index         int
	prompts       []string
	continuations []string
}

func (s *sequenceFixtureScorer) ScoreContinuations(
	_ context.Context, prompt string, continuations []string,
) ([]sequencescore.Score, error) {
	s.prompts = append(s.prompts, prompt)
	s.continuations = append(s.continuations, continuations...)
	score := s.scores[s.index%len(s.scores)]
	s.index++
	return []sequencescore.Score{score}, nil
}
