package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestATrainingRecipeNoVerdictCoversReadsNotApplicableAndFilesNoRow holds the
// owner's decision on the training recipes whose sessions only prove a route:
// their held-out quality is not applicable, with the reason that names how a
// verdict is published, and no unscored row is filed for it, while a task whose
// held-out evaluation is still proposed keeps filing one.
func TestATrainingRecipeNoVerdictCoversReadsNotApplicableAndFilesNoRow(t *testing.T) {
	t.Parallel()
	cell, others := qualityCell(recipe.TaskTraining, nil, nil)
	if cell.Measured() || !cell.NotApplicable || cell.Reason != trainingProvesRoute || len(others) != 0 {
		t.Fatalf("a training recipe without a verdict reads %+v (%d others), want not applicable for %q", cell, len(others), trainingProvesRoute)
	}
	if !strings.Contains(cell.Reason, "train -objective") || !strings.Contains(cell.Reason, "proves its route") {
		t.Fatalf("the reason %q does not name how a verdict is published and why this recipe has none", cell.Reason)
	}
	if proposed := heldoutProposals[recipe.TaskTraining]; proposed != "" {
		t.Fatalf("training still proposes %q, which would file a row its decision declared not applicable", proposed)
	}
	forecast, _ := qualityCell(recipe.TaskForecast, nil, nil)
	record := Record{Rows: []Row{
		{Name: "trained", Task: recipe.TaskTraining, Cells: map[metric.Name]Cell{metric.Quality: cell}},
		{Name: "forecaster", Task: recipe.TaskForecast, Cells: map[metric.Name]Cell{metric.Quality: forecast}},
	}}
	var titles []string
	for _, gap := range record.Gaps(nil) {
		if strings.Contains(gap.Title, "held-out evaluation") {
			titles = append(titles, gap.Title)
		}
	}
	if len(titles) != 1 || !strings.Contains(titles[0], "forecast") {
		t.Fatalf("held-out evaluation rows filed = %v, want only the forecast task, whose evaluation is still proposed", titles)
	}
}
