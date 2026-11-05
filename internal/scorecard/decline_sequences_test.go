package scorecard

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/sequencescore"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestGeneratedSlowQualityDecline reads one model's held-out quality at a row of commits, each of which loses five
// more of its hundred cases and gains none, and asks at each reading what the loop would file: Gaps against
// Reference, kept by Supported. A share that falls from 1.00 to 0.45 is filed, judged against the level the model
// held, not only against the reading before, where each step alone is within noise.
func TestGeneratedSlowQualityDecline(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	identity := modelrecipe.ProgramIdentity{
		Model:      testutil.ArtifactID(t, artifact.KindModel, "slow model"),
		Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, "slow definition"),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "slow recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/slow", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	testcheck.NoError(t, err)
	const cases, perReading, readings = 100, 5, 12
	suite := evaluation.MultipleChoiceSuite{Kind: evaluation.MultipleChoiceKind, Schema: "fixture/slow/v1", Source: "slow", Normalization: sequencescore.NormalizationSum, Aggregation: evaluation.AggregationAccuracy}
	for index := range cases {
		suite.Cases = append(suite.Cases, evaluation.MultipleChoiceCase{Name: fmt.Sprintf("case-%03d", index), Prompt: fmt.Sprintf("Question %d\nAnswer:", index), Candidates: []string{" A", " B"}})
	}
	source, err := json.Marshal(suite)
	testcheck.NoError(t, err)
	contract := Contract{Metric: shareMetric, Origin: originEvaluation, Suite: "slow", Protocol: "raw-completion", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "slow comparison")}
	row := func(quality Cell) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = quality
		return Row{Model: identity.Model, Name: "slow", Task: recipe.TaskInference, Cells: cells}
	}
	var records []Record
	for reading := range readings {
		commit := fmt.Sprintf("%x", reading+1)
		wrong := map[string]bool{}
		for _, testCase := range suite.Cases[:reading*perReading] {
			wrong[testCase.Prompt] = true
		}
		campaign, err := evaluation.NewCampaign(store, answerer{wrong}, identity, environment, strings.Repeat(commit, 40/len(commit)))
		testcheck.NoError(t, err)
		compiled, err := evaluation.CompileSuite(source, campaign.Authorities())
		testcheck.NoError(t, err)
		result, err := campaign.Evaluate(ctx, compiled)
		testcheck.NoError(t, err)
		value := float64(cases-reading*perReading) / cases
		cell := Cell{Value: value, Source: result.Evaluation, Direction: runrecord.DirectionMaximize, Cases: cases, Noise: shareError(value, cases), Contract: contract,
			Compared: "internal/fixture TestPairedPathMatchesItsReference"}
		records = append(records, Record{Commit: commit, Rows: []Row{row(cell)}})
	}
	testcheck.NoError(t, store.Close())
	filed := 0
	for reading := 1; reading < readings; reading++ {
		kept, dropped := Supported(ctx, root, records[reading].Gaps(Reference(records[:reading+1])))
		lost := uint64(reading * perReading)
		tail := tensorstats.ExactPairedBinary(lost, 0, cases-lost).Tail
		quality := 0
		for _, gap := range kept {
			if gap.quality != nil {
				quality++
			}
		}
		filed += quality
		if tail <= pairedTail && quality == 0 {
			t.Errorf("reading %d: %d cases lost since the first stand against it (tail %.2g), and no decline was filed", reading, lost, tail)
		}
		t.Logf("reading %2d: %3d of %d right, %2d lost since the first (tail %.2g, stands: %t); filed %d, dropped %d", reading, cases-int(lost), cases, lost, tail, tail <= pairedTail, quality, len(dropped))
	}
	if filed == 0 {
		t.Errorf("a share that fell from 1.00 to %.2f was never filed", float64(cases-(readings-1)*perReading)/cases)
	}
}
