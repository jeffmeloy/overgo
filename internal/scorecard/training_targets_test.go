package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestADeclaredTrainingTargetIsARowUntilAVerdictMeetsIt: a declared target files a row of its own for a model the record
// trains, and it stays open through no verdict, a gain short of the target, a gain its cases do not support and a verdict on
// another objective. A supported verdict that gains what it asks closes it. A model the record does not train files nothing.
func TestADeclaredTrainingTargetIsARowUntilAVerdictMeetsIt(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "trained")
	objective := testutil.ArtifactID(t, artifact.KindEvidence, "objective")
	target := TrainingTarget{
		Model: model, Name: "small", Alias: "objective.registered.text", Objective: objective,
		Steps: 128, Sequence: 512, Order: 25, MaxWall: "10m", Gain: 0.01,
	}
	base := 0.5
	row := func(cell Cell) Row {
		return Row{Model: model, Name: "small", Task: recipe.TaskTraining, Cells: map[metric.Name]Cell{metric.Quality: cell}}
	}
	verdict := func(value float64, supported bool, judged artifact.ID) Cell {
		return Cell{
			Value: value, Base: &base, Supported: supported, Objective: judged, Direction: runrecord.DirectionMaximize,
			Source: testutil.ArtifactID(t, artifact.KindEvaluation, "verdict"),
		}
	}
	other := testutil.ArtifactID(t, artifact.KindEvidence, "another objective")
	for name, cell := range map[string]Cell{
		"no verdict":        gap("no verdict"),
		"short of it":       verdict(0.505, true, objective),
		"unsupported":       verdict(0.52, false, objective),
		"another objective": verdict(0.52, true, other),
	} {
		gaps := Record{Rows: []Row{row(cell)}}.trainingGaps([]TrainingTarget{target})
		if len(gaps) != 1 || !KnownGapKey(gaps[0].Key) || !strings.Contains(gaps[0].Title, "Train small") ||
			!strings.Contains(gaps[0].Detail, "-objective objective.registered.text") || !strings.Contains(gaps[0].Detail, "-steps 128 -seq 512 -order-seed 25") {
			t.Errorf("%s: the target files %+v, want its row", name, gaps)
		}
	}
	if gaps := (Record{Rows: []Row{row(verdict(0.52, true, objective))}}).trainingGaps([]TrainingTarget{target}); len(gaps) != 0 {
		t.Errorf("a supported verdict that meets the target leaves %+v", gaps)
	}
	served := row(gap("no verdict"))
	served.Task = recipe.TaskInference
	if gaps := (Record{Rows: []Row{served}}).trainingGaps([]TrainingTarget{target}); len(gaps) != 0 {
		t.Errorf("a model the record does not train files %+v", gaps)
	}
	twoModels := target
	twoModels.Model = testutil.ArtifactID(t, artifact.KindModel, "another model")
	if target.key() == twoModels.key() {
		t.Error("two models' targets share a row")
	}
}

// TestATrainingTargetARunCouldNotStartFromIsRefused: a repeated target, one of no gain, and one whose bounded run names no
// order, which would read only the first articles of its data.
func TestATrainingTargetARunCouldNotStartFromIsRefused(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "trained")
	objective := testutil.ArtifactID(t, artifact.KindEvidence, "objective")
	whole := `{"model":"` + model.String() + `","name":"n","alias":"a","objective":"` + objective.String() + `","steps":1,"sequence":1,"order":1,"max_wall":"1m","gain":0.1}`
	if _, err := parseTargets([]byte(`{"targets":[` + whole + `]}`)); err != nil {
		t.Fatalf("a whole target: %v", err)
	}
	if _, err := parseTargets([]byte(`{"targets":[` + whole + `,` + whole + `]}`)); err == nil {
		t.Error("a repeated target parses")
	}
	if _, err := parseTargets([]byte(strings.Replace(`{"targets":[`+whole+`]}`, `"gain":0.1`, `"gain":0`, 1))); err == nil {
		t.Error("a target of no gain parses")
	}
	if _, err := parseTargets([]byte(strings.Replace(`{"targets":[`+whole+`]}`, `"order":1`, `"order":0`, 1))); err == nil {
		t.Error("a target with no order parses")
	}
}
