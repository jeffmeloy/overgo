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
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
)

// scoreChoicesAlone answers and scores a choice suite in a store of its own,
// where nothing has been answered before.
func scoreChoicesAlone(t *testing.T, ctx context.Context, scorer continuationScorer, suite MultipleChoiceSuite) ([]choiceObservation, float64, error) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	plan := Plan{body: planBody{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		Execution: planID(t, artifact.KindProfile, "execution"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
	}}
	observations, _, accuracy, err := scoreMultipleChoice(ctx, store, plan, scorer, suite)
	return observations, accuracy, err
}

// choiceTally is a model that scores a case's candidates from a table by the
// case's prompt, and keeps each prompt it is put. Put the prompt stop, it
// fails, once.
type choiceTally struct {
	observedRuntime
	scores map[string][]sequencescore.Score
	asked  []string
	stop   string
}

var errTallyStopped = errors.New("the fixture model stopped")

func (tally *choiceTally) ScoreContinuations(_ context.Context, prompt string, _ []string) ([]sequencescore.Score, error) {
	if prompt == tally.stop {
		tally.stop = ""
		return nil, errTallyStopped
	}
	tally.asked = append(tally.asked, prompt)
	return slices.Clone(tally.scores[prompt]), nil
}

// TestScoringFixRunsNoCaseAgain holds an evaluation to answers written once.
//
// Two models answer a suite of two-candidate cases, one more than a record of
// answers holds. Scored by summed likelihood, the candidate model reads below
// the base, and the decision recorded against the base refuses it. The scorer
// is then changed, to the mean a token, and then the answer one case is
// scored against: each evaluation puts no case to either model again, reads
// the records of answers the first wrote and writes none, and scores them
// anew, and the decision against the base, now the base's reading under the
// new scorer, selects the candidate. The evidence of each evaluation names its
// run and its evaluation record and copies neither's measurements, and reads
// back with theirs.
//
// A model that stops part way keeps the answers it gave: the evaluation asked
// again puts it only the cases after them.
func TestScoringFixRunsNoCaseAgain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const cases = observedPending + 8
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/scoring-fix/v1", Source: "scoring-fix",
		Normalization: sequencescore.NormalizationSum, Aggregation: AggregationAccuracy,
	}
	// The base is right on every other case under either scorer. The candidate's
	// right answer is the longer one each time: likelier a token, and less likely summed.
	base, candidate := map[string][]sequencescore.Score{}, map[string][]sequencescore.Score{}
	for index := range cases {
		prompt := fmt.Sprintf("Question %d\nAnswer:", index)
		suite.Cases = append(suite.Cases, MultipleChoiceCase{Name: fmt.Sprintf("case-%02d", index), Prompt: prompt, Candidates: []string{" A", " B"}})
		base[prompt] = []sequencescore.Score{{LogProbability: -2, Tokens: 2}, {LogProbability: -3, Tokens: 1}}
		if index%2 == 1 {
			base[prompt][0].LogProbability = -8
		}
		candidate[prompt] = []sequencescore.Score{{LogProbability: -4, Tokens: 4}, {LogProbability: -3, Tokens: 1}}
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	type model struct {
		tally      *choiceTally
		campaign   *Campaign
		definition recipe.Definition
	}
	served := func(name string, scores map[string][]sequencescore.Score) model {
		definition := publishSelectionModel(t, store, name, 4096)
		tally := &choiceTally{scores: scores}
		campaign, err := newCampaign(store, tally, modelrecipe.ProgramIdentity{
			Model: planID(t, artifact.KindModel, name+"-model"), Definition: planID(t, artifact.KindModelDefinition, name+"-definition"), Recipe: definition.ID,
		}, environment, runtimeLedgerCommit, LifecycleIsolated)
		if err != nil {
			t.Fatal(err)
		}
		return model{tally, campaign, definition}
	}
	// evaluated evaluates the suite as it now stands, and returns the result with the records of answers the store then holds for it.
	evaluated := func(subject model) (CampaignResult, []artifact.ID, error) {
		source, err := json.Marshal(suite)
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := CompileSuite(source, subject.campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, evaluateErr := subject.campaign.Evaluate(ctx, compiled)
		asked := make([]askedChoice, len(suite.Cases))
		for index, testCase := range suite.Cases {
			asked[index] = askedChoice{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates}
		}
		observed, err := compiled.Plan().observed(asked)
		if err != nil {
			t.Fatal(err)
		}
		edges, err := store.Children(ctx, observed)
		if err != nil {
			t.Fatal(err)
		}
		var records []artifact.ID
		for _, edge := range edges {
			if edge.Child.Kind() == artifact.KindOutput {
				records = append(records, edge.Child)
			}
		}
		return result, records, evaluateErr
	}
	accuracy := func(result CampaignResult) float64 {
		value, _, measured := admittedMetric(result.Metrics, "accuracy")
		if !measured {
			t.Fatalf("the evaluation measured %v, and no accuracy", result.Metrics)
		}
		return value
	}

	incumbent, challenger := served("incumbent", base), served("challenger", candidate)
	baseline, baseRecords, err := evaluated(incumbent)
	if err != nil {
		t.Fatal(err)
	}
	first, records, err := evaluated(challenger)
	if err != nil {
		t.Fatal(err)
	}
	if len(incumbent.tally.asked) != cases || len(challenger.tally.asked) != cases || len(records) != 2 || len(baseRecords) != 2 ||
		accuracy(baseline) != 0.5 || accuracy(first) != 0 {
		t.Fatalf("scored by summed likelihood the base reads %v and the candidate %v, having been put %d and %d cases and written %d and %d records of answers; want 0.5 and 0, %d cases each, and two records each",
			accuracy(baseline), accuracy(first), len(incumbent.tally.asked), len(challenger.tally.asked), len(baseRecords), len(records), cases)
	}
	if _, err := SelectServingRecipe(ctx, store, recipe.TaskInference, "accuracy", baseline.Evidence, []artifact.ID{first.Evidence}); err == nil || !strings.Contains(err.Error(), "meets the routing threshold") {
		t.Fatalf("a candidate below its base = %v, want it refused", err)
	}

	// The scorer changes, and then the answer one case is scored against.
	for _, fix := range []struct {
		name            string
		change          func()
		base, candidate float64
	}{
		{"scored by the mean a token", func() { suite.Normalization = sequencescore.NormalizationMean }, 0.5, 1},
		{"with one case's answer corrected", func() { suite.Cases[0].Answer = 1 }, float64(cases/2-1) / cases, float64(cases-1) / cases},
	} {
		fix.change()
		rebased, rebasedRecords, err := evaluated(incumbent)
		if err != nil {
			t.Fatal(err)
		}
		rescored, rescoredRecords, err := evaluated(challenger)
		if err != nil {
			t.Fatal(err)
		}
		if len(incumbent.tally.asked) != cases || len(challenger.tally.asked) != cases {
			t.Fatalf("%s, the models were put %d and %d more cases, want none", fix.name, len(incumbent.tally.asked)-cases, len(challenger.tally.asked)-cases)
		}
		if !slices.Equal(rebasedRecords, baseRecords) || !slices.Equal(rescoredRecords, records) || rescored.Report == first.Report {
			t.Fatalf("%s, the evaluations read records %v and %v of answers and reported %s; want the records the first evaluations wrote, %v and %v, and another report than %s",
				fix.name, rebasedRecords, rescoredRecords, rescored.Report, baseRecords, records, first.Report)
		}
		if accuracy(rebased) != fix.base || accuracy(rescored) != fix.candidate {
			t.Fatalf("%s the base reads %v and the candidate %v, want %v and %v", fix.name, accuracy(rebased), accuracy(rescored), fix.base, fix.candidate)
		}
		decision, err := SelectServingRecipe(ctx, store, recipe.TaskInference, "accuracy", rebased.Evidence, []artifact.ID{rescored.Evidence})
		if err != nil || decision.Selected != challenger.definition.ID || decision.Signal.ThresholdEvidence != rebased.Evidence || decision.Signal.QualityThreshold != fix.base {
			t.Fatalf("%s the decision against the base = (%s over a threshold of %v, %v), want the candidate's recipe %s over the base's reading %v",
				fix.name, decision.Selected, decision.Signal.QualityThreshold, err, challenger.definition.ID, fix.base)
		}
		// The evidence names its run and its evaluation record, and copies neither's measurements.
		content, found, err := artifact.ReadContent(ctx, store, rescored.Evidence)
		if err != nil || !found || bytes.Contains(content.Data, []byte(`"metrics"`)) || bytes.Contains(content.Data, []byte(`"phases"`)) {
			t.Fatalf("%s the evidence document is %s (found %t, %v), want it to copy no metric and no phase", fix.name, content.Data, found, err)
		}
		evidence, err := RequireEvaluationEvidence(ctx, store, rescored.Evidence)
		record, recordErr := runrecord.RequireEvaluation(ctx, store, rescored.Evaluation)
		if err != nil || recordErr != nil || !slices.Equal(evidence.Metrics, record.Metrics) || len(evidence.Phases) == 0 {
			t.Fatalf("%s the evidence reads back with metrics %v and phases %v (%v, %v), want its evaluation record's %v and its run's", fix.name, evidence.Metrics, evidence.Phases, err, recordErr, record.Metrics)
		}
	}

	// A model that stops keeps what it answered, past the last full record of answers too.
	const stopped = observedPending + 3
	interrupted := served("interrupted", candidate)
	interrupted.tally.stop = suite.Cases[stopped].Prompt
	if _, records, err := evaluated(interrupted); !errors.Is(err, errTallyStopped) || len(interrupted.tally.asked) != stopped || len(records) != 2 {
		t.Fatalf("a model that stops at case %d = %v, having answered %d cases in %d records; want its error, and each case it answered written down", stopped, err, len(interrupted.tally.asked), len(records))
	}
	resumed, _, err := evaluated(interrupted)
	if err != nil || len(interrupted.tally.asked) != cases || !slices.Equal(interrupted.tally.asked[stopped:], func() (rest []string) {
		for _, testCase := range suite.Cases[stopped:] {
			rest = append(rest, testCase.Prompt)
		}
		return rest
	}()) || accuracy(resumed) != float64(cases-1)/cases {
		t.Fatalf("asked again, the model was put %d cases in all and reads %v (%v); want the %d it had not answered and no other, and the candidate's reading", len(interrupted.tally.asked), accuracy(resumed), err, cases-stopped)
	}
}
