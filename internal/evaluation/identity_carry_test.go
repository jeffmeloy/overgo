package evaluation

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
)

// carriedModel is a model's type in this package, which the fixture's build
// counts an identity for.
type carriedModel struct{}

// TestAnswersOutliveOnlyTheCommitsThatLeaveTheirCodeUnchanged evaluates one
// suite over one store at commit after commit, as measurement runs do, and
// counts the cases put to the model each time.
//
// Under a build that counted an identity for the model's package, the first
// evaluation puts every case. A later commit under the same identity puts
// none: its run is a succeeded run at that commit whose report is a full
// report of that commit's plan, the same readings, which the evidence index
// reads as that commit's and as nothing carried. A commit that moves the
// identity puts every case again; the same commit again puts none; and a
// commit that returns to the first identity puts none, the answers given
// under it being the store's still. Another environment puts every case.
//
// A build that counted nothing keeps its answers under its commit, as every
// build did: each new commit puts every case, and the same commit again none.
// So does a campaign that was never told what answers its cases, and one
// whose model is hosted, whatever its build counted: what answers there is
// not the code.
func TestAnswersOutliveOnlyTheCommitsThatLeaveTheirCodeUnchanged(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const cases = observedPending + 3
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/identity-carry/v1", Source: "store/identity-carry",
		Normalization: sequencescore.NormalizationSum, Aggregation: AggregationAccuracy,
	}
	scores := map[string][]sequencescore.Score{}
	for index := range cases {
		prompt := fmt.Sprintf("Question %d\nAnswer:", index)
		suite.Cases = append(suite.Cases, MultipleChoiceCase{Name: fmt.Sprintf("case-%02d", index), Prompt: prompt, Candidates: []string{" A", " B"}, Answer: index % 2})
		scores[prompt] = []sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}
	}
	source, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "carry model"),
		Definition: planID(t, artifact.KindModelDefinition, "carry definition"),
		Recipe:     planID(t, artifact.KindRecipe, "carry recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/carry", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	platform := func(backend, driver string) runrecord.Environment {
		t.Helper()
		value, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: backend, Driver: driver})
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	device, other, hosted := platform("fixture", "fixture"), platform("fixture", "other"), platform(runrecord.BackendRemote, "fixture")

	// counted is what the build of each evaluation counted for the model's package: nothing, when empty.
	counted := ""
	build := func(pkg string) (string, bool) {
		return counted, counted != "" && pkg == reflect.TypeFor[carriedModel]().PkgPath()
	}
	// evaluate evaluates the suite at a commit and says how many cases it put to the model.
	evaluate := func(commit string, environment runrecord.Environment, told bool) (CampaignResult, CompiledSuite, int) {
		t.Helper()
		tally := &choiceTally{scores: scores}
		campaign, err := newCampaign(store, tally, identity, environment, strings.Repeat(commit, 40), LifecycleIsolated)
		if err != nil {
			t.Fatal(err)
		}
		if told {
			campaign.answeredBy(reflect.TypeFor[carriedModel]().PkgPath(), build)
		}
		compiled, err := CompileSuite(source, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, err := campaign.Evaluate(ctx, compiled)
		if err != nil {
			t.Fatal(err)
		}
		return result, compiled, len(tally.asked)
	}

	counted = "first"
	first, firstSuite, asked := evaluate("a", device, true)
	if asked != cases {
		t.Fatalf("the first evaluation put %d of %d cases to the model", asked, cases)
	}
	second, secondSuite, asked := evaluate("b", device, true)
	if asked != 0 || !slices.Equal(second.Metrics, first.Metrics) {
		t.Fatalf("a commit that left the identity where it was put %d cases to the model and read %v; want none, and the first evaluation's %v", asked, second.Metrics, first.Metrics)
	}
	if secondSuite.Plan().Identity() == firstSuite.Plan().Identity() || second.Report == first.Report || second.Run == first.Run {
		t.Fatalf("the second commit's evaluation is the first's: plan %s, report %s, run %s", secondSuite.Plan().Identity(), second.Report, second.Run)
	}
	run, err := runrecord.RequireRun(ctx, store, second.Run)
	if err != nil || run.CodeCommit != strings.Repeat("b", 40) || run.Outcome != runrecord.OutcomeSucceeded || !slices.Equal(run.Outputs, []artifact.ID{second.Report}) {
		t.Fatalf("the second commit's run is not a succeeded run at that commit reporting its report: %+v %v", run, err)
	}
	if descriptor, found, err := store.Artifact(ctx, second.Report); err != nil || !found || descriptor.Schema == ReplayReceiptSchema {
		t.Fatalf("the second commit's report is %+v (found %t, %v); want a full report and no receipt of a replay", descriptor, found, err)
	}
	if _, err := RequireEvaluationEvidence(ctx, store, second.Evidence); err != nil {
		t.Fatalf("the second commit's evidence does not verify: %v", err)
	}
	summaries := LatestEvidence(ctx, store, map[artifact.ID]string{secondSuite.Descriptor().Dataset: suite.Source}, 0).EvaluationsByRecipe[identity.Recipe]
	if len(summaries) != 1 || summaries[0].Record != second.Evaluation || summaries[0].CodeCommit != strings.Repeat("b", 40) || summaries[0].Carried.Valid() || summaries[0].ScoredCommit != "" {
		t.Fatalf("the evidence index reads %+v; want the second commit's record alone, as that commit's and as nothing carried", summaries)
	}

	counted = "second"
	if _, _, asked := evaluate("c", device, true); asked != cases {
		t.Fatalf("a commit that moved the identity put %d cases to the model; want all %d", asked, cases)
	}
	if _, _, asked := evaluate("c", device, true); asked != 0 {
		t.Fatalf("the same commit again put %d cases to the model; want none", asked)
	}
	counted = "first"
	if _, _, asked := evaluate("d", device, true); asked != 0 {
		t.Fatalf("a commit that returned to the first identity put %d cases to the model; want none", asked)
	}
	if _, _, asked := evaluate("d", other, true); asked != cases {
		t.Fatalf("another environment put %d cases to the model; want all %d", asked, cases)
	}

	// A build that counted nothing, a campaign told nothing, and a hosted model keep their answers under the commit.
	for _, kept := range []struct {
		what        string
		counted     string
		environment runrecord.Environment
		told        bool
		commits     [2]string
	}{
		{"a build that counted nothing", "", device, true, [2]string{"e", "f"}},
		{"a campaign that was not told what answers its cases", "first", device, false, [2]string{"1", "2"}},
		{"a hosted model", "first", hosted, true, [2]string{"3", "4"}},
	} {
		counted = kept.counted
		for _, commit := range kept.commits {
			if _, _, asked := evaluate(commit, kept.environment, kept.told); asked != cases {
				t.Fatalf("%s, at a new commit, put %d cases to the model; want all %d", kept.what, asked, cases)
			}
		}
		if _, _, asked := evaluate(kept.commits[1], kept.environment, kept.told); asked != 0 {
			t.Fatalf("%s, at the same commit again, put %d cases to the model; want none", kept.what, asked)
		}
	}
}
