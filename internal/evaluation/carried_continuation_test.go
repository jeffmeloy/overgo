package evaluation

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// wordMemory is a model whose deferred turn costs a document's word count
// with a fresh memory and a quarter of it with the memory the document wrote,
// and which says what it was asked.
type wordMemory struct{ asked *[]string }

func (m wordMemory) ScoreCarried(_ context.Context, text string, chunks, opening int) (carried, reset sequencescore.Score, err error) {
	words := float64(len(strings.Fields(text)))
	*m.asked = append(*m.asked, strings.Fields(text)[0])
	if chunks != 4 || opening != 2 {
		return carried, reset, errors.New("asked for other chunks or another opening than the suite's")
	}
	return sequencescore.Score{LogProbability: -words / 4, Tokens: uint64(opening)}, sequencescore.Score{LogProbability: -words, Tokens: uint64(opening)}, nil
}

func (wordMemory) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the memory measure generates no text")
}

func (wordMemory) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("the memory measure scores no continuation")
}

// TestCarriedContinuationScoresWhatAMemoryCarried holds the measure of a model
// that remembers a document in notes it writes to itself. The model is asked,
// for each document, the deferred turn after the suite's chunks with the
// suite's opening; the report gives each document's two losses a token, the
// mean of each over the documents, the gain, which is the loss with a fresh
// memory less the loss with the one the document wrote, and the standard error
// of the gain over the documents. The gain is the metric read first. A runtime
// that carries no memory is refused, and so is a suite with no chunks, no
// opening, no text, or a case named twice.
func TestCarriedContinuationScoresWhatAMemoryCarried(t *testing.T) {
	t.Parallel()
	suite := CarriedContinuationSuite{Kind: carriedContinuationKind, Schema: "fixture/v1", Source: "fixture", Chunks: 4, Opening: 2, Cases: []CarriedContinuationCase{
		{Name: "eight words", Text: "one two three four five six seven eight"},
		{Name: "sixteen words", Text: "a b c d e f g h i j k l m n o p"},
	}}
	compiled, err := compileCarriedContinuation(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindCarriedContinuation(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	var asked []string
	report, err := evaluateCarriedContinuation(t.Context(), store, wordMemory{&asked}, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	// A token, the first document loses 4 fresh and 1 carried, the second 8 and
	// 2: gains of 3 and 6, whose mean is 4.5 and whose standard error is 1.5.
	near := func(got, want float64) bool { return math.Abs(got-want) < 1e-12 }
	if len(report.Observations) != 2 || !near(report.Observations[0].ResetLoss, 4) || !near(report.Observations[0].CarriedLoss, 1) ||
		!near(report.ResetLoss, 6) || !near(report.CarriedLoss, 1.5) || !near(report.CarriedGain, 4.5) || !near(report.GainError, 1.5) {
		t.Fatalf("report = %+v, want losses of 6 fresh and 1.5 carried and a gain of 4.5 with a standard error of 1.5", report)
	}
	if strings.Join(asked, " ") != "one a" {
		t.Fatalf("the model was asked about %q, want each document once in the suite's order", asked)
	}
	metrics := map[string]float64{}
	for _, metric := range carriedContinuationMetrics(report.CarriedGain, report.CarriedLoss, report.ResetLoss) {
		metrics[metric.Name] = metric.Value
	}
	if name, value, ok := HeadlineMetric(metrics); !ok || name != "carried_gain" || !near(value, 4.5) {
		t.Fatalf("the metric read first is %s = %v, want the gain", name, value)
	}
	if _, err := evaluateCarriedContinuation(t.Context(), store, textOnlyRuntime{}, compiled, plan); err == nil {
		t.Fatal("a runtime that carries no memory was evaluated")
	}
	for name, change := range map[string]func(*CarriedContinuationSuite){
		"no chunks":       func(s *CarriedContinuationSuite) { s.Chunks = 0 },
		"no opening":      func(s *CarriedContinuationSuite) { s.Opening = 0 },
		"no text":         func(s *CarriedContinuationSuite) { s.Cases[0].Text = " " },
		"a repeated case": func(s *CarriedContinuationSuite) { s.Cases[1].Name = s.Cases[0].Name },
		"no cases":        func(s *CarriedContinuationSuite) { s.Cases = nil },
	} {
		broken := suite
		broken.Cases = append([]CarriedContinuationCase(nil), suite.Cases...)
		change(&broken)
		if _, err := compileCarriedContinuation(broken); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
}

// TestWikiTextDocumentsAreTheStartsOfLongArticles holds the builder to the
// articles of a split: an article opens at the line that titles it and not at
// a section's; one too short to hold the chunks and the opening is left out;
// one long enough gives its start, cut to the builder's length and ended on a
// word; and the suite says how many chunks are written and how many tokens are
// asked for.
func TestWikiTextDocumentsAreTheStartsOfLongArticles(t *testing.T) {
	t.Parallel()
	sentence := " The quick brown fox jumps over the lazy dog , and the dog does not mind at all . \n"
	long := strings.Repeat(sentence, documentCharacters/len(sentence)+2)
	lines := []string{
		"", " = A Short Article = \n", "", sentence,
		" = A Long Article = \n", "", long, " = = A Section = = \n", sentence,
		" = Another Long One = \n", long,
	}
	built, count, err := BuildWikiTextDocuments(lines)
	if err != nil {
		t.Fatal(err)
	}
	suite := built.(CarriedContinuationSuite)
	if count != 2 || len(suite.Cases) != 2 || suite.Chunks != carriedChunks || suite.Opening != carriedOpening || suite.Kind != carriedContinuationKind {
		t.Fatalf("the suite holds %d documents of %d chunks and an opening of %d, want the two long articles", count, suite.Chunks, suite.Opening)
	}
	for index, article := range []string{" = A Long Article = \n" + long + " = = A Section = = \n" + sentence, " = Another Long One = \n" + long} {
		text := suite.Cases[index].Text
		if !strings.HasPrefix(article, text) || len(text) > documentCharacters || len(text) < documentCharacters-len(sentence) || article[len(text)] != ' ' {
			t.Fatalf("document %d runs to %d characters and ends %q, want the start of its article ended on a word", index, len(text), text[len(text)-12:])
		}
	}
	if strings.Contains(suite.Cases[0].Text, "Another Long One") {
		t.Fatal("a document runs into the article after it")
	}
	if _, _, err := BuildWikiTextDocuments(lines[:4]); err == nil {
		t.Fatal("a suite was built from a split with no article long enough")
	}
}
