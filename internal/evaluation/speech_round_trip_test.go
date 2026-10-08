package evaluation

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// echoSpeaker is a speech session whose judge hears exactly what a script says.
type echoSpeaker struct {
	heard map[string]string
	judge string
}

func (s echoSpeaker) RoundTrip(_ context.Context, text, _ string, _ int64) (string, error) {
	heard, known := s.heard[text]
	if !known {
		return "", errors.New("no script for the text")
	}
	return heard, nil
}

func (s echoSpeaker) Judge() string { return s.judge }

func (echoSpeaker) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("a speech session generates no text")
}

func (echoSpeaker) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("a speech session scores no continuation")
}

// TestSpeechRoundTripScoresTheWordsThatSurviveTheTrip holds the report to the
// edit distance of what the judge heard to the text, over the texts' words, with
// the capitals and punctuation of the text ignored, and refuses a session whose
// judge is not the suite's.
func TestSpeechRoundTripScoresTheWordsThatSurviveTheTrip(t *testing.T) {
	t.Parallel()
	suite := SpeechRoundTripSuite{Kind: speechRoundTripKind, Schema: "fixture/v1", Source: "fixture", Voice: "alba", Seed: 7, Judge: "recognizer-a", Cases: []SpeechRoundTripCase{
		{Name: "exact", Text: "He hoped there would be stew."},
		{Name: "two wrong", Text: "The cat sat on the mat."},
	}}
	compiled, err := compileSpeechRoundTrip(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindSpeechRoundTrip(compiled, ExactAuthorities{
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
	speaker := echoSpeaker{judge: "recognizer-a", heard: map[string]string{
		"He hoped there would be stew.": "he hoped there would be stew",
		"The cat sat on the mat.":       "the bat sat on mat",
	}}
	report, err := evaluateSpeechRoundTrip(t.Context(), store, speaker, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	// The first case is exact; the second, six words, loses "cat" to "bat" and the second "the".
	if report.Words != 12 || report.Errors != 2 || report.WordErrorRate != 2.0/12.0 ||
		report.Observations[0].Errors != 0 || report.Observations[1].Errors != 2 {
		t.Fatalf("report = %+v, want 2 errors in 12 words", report)
	}
	other := speaker
	other.judge = "recognizer-b"
	if _, err := evaluateSpeechRoundTrip(t.Context(), store, other, compiled, plan); err == nil {
		t.Fatal("a session judged by another recognizer was evaluated")
	}
	if _, err := evaluateSpeechRoundTrip(t.Context(), store, textOnlyRuntime{}, compiled, plan); err == nil {
		t.Fatal("a runtime that cannot speak was evaluated")
	}
}

// TestSpeechRoundTripCountsWordsWrittenOneWay holds a round-trip suite to the
// counting it names. A judge that writes digits for a spelled-out number and
// a contraction in full is charged for each by the plain steps, and for none
// by the recognizers' own normalizer; a suite that names no counting is the
// suite it always was, to its identity, and is counted the plain way; a suite
// built now names the normalizer; and no suite names a step the evaluator
// does not own.
func TestSpeechRoundTripCountsWordsWrittenOneWay(t *testing.T) {
	t.Parallel()
	const (
		said  = "Departing from five hundred thousand throats, three cheers burst forth; it wasn't the goat's."
		heard = "departing from 500000 throats 3 cheers burst forth it was not the goat is"
	)
	score := func(suite SpeechRoundTripSuite) (speechRoundTripPlan, speechRoundTripReport) {
		t.Helper()
		compiled, err := compileSpeechRoundTrip(suite)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := bindSpeechRoundTrip(compiled, ExactAuthorities{
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
		report, err := evaluateSpeechRoundTrip(t.Context(), store, echoSpeaker{judge: "recognizer-a", heard: map[string]string{said: heard}}, compiled, plan)
		if err != nil {
			t.Fatal(err)
		}
		return compiled, report
	}
	plain := SpeechRoundTripSuite{Kind: speechRoundTripKind, Schema: "fixture/v1", Source: "fixture", Voice: "alba", Seed: 7, Judge: "recognizer-a", Cases: []SpeechRoundTripCase{{Name: "numbers", Text: said}}}
	oneWay := plain
	oneWay.Normalization = slices.Clone(writtenOneWay)
	plainPlan, plainReport := score(plain)
	oneWayPlan, oneWayReport := score(oneWay)
	if plainReport.Errors == 0 {
		t.Fatal("the plain steps charged a judge nothing for writing in its own conventions: the case tries nothing")
	}
	if oneWayReport.Errors != 0 || oneWayReport.Words != uint64(len(strings.Fields(heard))) {
		t.Fatalf("written one way, the judge is charged %d of %d words, and it wrote what was said", oneWayReport.Errors, oneWayReport.Words)
	}
	if plainPlan.identity == oneWayPlan.identity {
		t.Fatal("a suite counted another way has the identity of the suite it was made from")
	}
	named := plain
	named.Normalization = slices.Clone(roundTripNormalization)
	if _, namedReport := score(named); namedReport.Errors != plainReport.Errors || namedReport.Words != plainReport.Words {
		t.Fatalf("a suite that names no counting is charged %d of %d words, and one that names the plain steps %d of %d", plainReport.Errors, plainReport.Words, namedReport.Errors, namedReport.Words)
	}
	if encoded, err := json.Marshal(plain); err != nil || bytes.Contains(encoded, []byte("normalization")) {
		t.Fatalf("a suite that names no counting is written with one, so a stored suite's identity would move: %s, %v", encoded, err)
	}
	built, err := speechRoundTripSuiteOf([]string{sentence("ONE TEXT OF THE CORPUS READ ALOUD")}, SpeechRoundTripSpec{Parquet: "corpus.parquet", Cases: 1, Voice: "alba", Judge: "granite"})
	if err != nil || !slices.Equal(built.Normalization, writtenOneWay) {
		t.Fatalf("a suite built now is counted by %v, %v", built.Normalization, err)
	}
	unknown := plain
	unknown.Normalization = []TranscriptionNormalization{"as-heard"}
	if _, err := compileSpeechRoundTrip(unknown); err == nil {
		t.Fatal("a suite named a counting the evaluator does not own")
	}
}

// TestSpeechRoundTripRefusesWhatItCannotCompile holds the compiler to a suite
// with no voice, no judge, no words or a repeated case.
func TestSpeechRoundTripRefusesWhatItCannotCompile(t *testing.T) {
	t.Parallel()
	valid := func() SpeechRoundTripSuite {
		return SpeechRoundTripSuite{Kind: speechRoundTripKind, Schema: "fixture/v1", Source: "fixture", Voice: "alba", Judge: "j", Cases: []SpeechRoundTripCase{
			{Name: "a", Text: "One two."}, {Name: "b", Text: "Three four."},
		}}
	}
	if _, err := compileSpeechRoundTrip(valid()); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*SpeechRoundTripSuite){
		"no voice":        func(s *SpeechRoundTripSuite) { s.Voice = "" },
		"no judge":        func(s *SpeechRoundTripSuite) { s.Judge = "" },
		"a text of marks": func(s *SpeechRoundTripSuite) { s.Cases[0].Text = " ... " },
		"a repeated case": func(s *SpeechRoundTripSuite) { s.Cases[1].Name = "a" },
		"no cases":        func(s *SpeechRoundTripSuite) { s.Cases = nil },
	} {
		suite := valid()
		change(&suite)
		if _, err := compileSpeechRoundTrip(suite); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
}

// TestSpeechRoundTripSuiteIsPickedBySeedFromTheUsableTexts holds the selection
// to its spec: the same seed picks the same texts, another seed others, each
// read as a sentence, and too few texts is refused.
func TestSpeechRoundTripSuiteIsPickedBySeedFromTheUsableTexts(t *testing.T) {
	t.Parallel()
	var texts []string
	for index := range 20 {
		texts = append(texts, sentence(fmt.Sprintf("TEXT NUMBER %d OF THE CORPUS READ ALOUD", index)))
	}
	spec := SpeechRoundTripSpec{Parquet: "corpus.parquet", Cases: 5, Seed: 17, Voice: "alba", SynthesisSeed: 7, Judge: "granite"}
	first, err := speechRoundTripSuiteOf(texts, spec)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := speechRoundTripSuiteOf(texts, spec)
	spec.Seed++
	other, _ := speechRoundTripSuiteOf(texts, spec)
	if len(first.Cases) != 5 || !slices.Equal(caseTexts(first), caseTexts(again)) || slices.Equal(caseTexts(first), caseTexts(other)) {
		t.Fatalf("selection is not a function of the seed: %v / %v / %v", caseTexts(first), caseTexts(again), caseTexts(other))
	}
	if _, err := compileSpeechRoundTrip(first); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	if got := sentence("HE HOPED THERE WOULD BE STEW"); got != "He hoped there would be stew." {
		t.Fatalf("sentence = %q", got)
	}
	spec.Cases = 21
	if _, err := speechRoundTripSuiteOf(texts, spec); err == nil || !strings.Contains(err.Error(), "fewer than the 21") {
		t.Fatalf("asking for more texts than there are: %v", err)
	}
}

func caseTexts(suite SpeechRoundTripSuite) []string {
	var texts []string
	for _, testCase := range suite.Cases {
		texts = append(texts, testCase.Text)
	}
	return texts
}
