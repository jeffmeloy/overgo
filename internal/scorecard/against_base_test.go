package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestATrainedModelsRowReadsAgainstItsBase: a trained model's row names the base it was trained from and holds, beside each
// cell both rows measured, the base's reading on the same measure: its raw-completion reading beside a raw-completion
// reading, though the base's headline is another protocol's. It is shown and never judged: a trained model below its base files no row
// that trains it, since the owner chooses what runs and what training starts from. A row no training made is read alone.
func TestATrainedModelsRowReadsAgainstItsBase(t *testing.T) {
	t.Parallel()
	weights := testutil.ArtifactID(t, artifact.KindModel, "the base's weights file, which training reads")
	base := testutil.ArtifactID(t, artifact.KindModel, "the base's directory, which serves")
	trained := testutil.ArtifactID(t, artifact.KindModel, "trained")
	source := testutil.ArtifactID(t, artifact.KindEvaluation, "reading")
	measured := func(value float64) Cell { return Cell{Value: value, Source: source} }
	read := func(value float64, protocol string) Cell {
		return Cell{Value: value, Source: source, Contract: Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu-pro", Protocol: protocol}}
	}
	rows := []Row{
		{Model: base, Name: "Qwen2.5-0.5B", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{
			metric.Quality: read(0.1042, "chat-template"), metric.Throughput: measured(505.7), metric.FirstOutput: measured(59.7),
		}, Heldout: []Cell{read(0.1696, "raw-completion")}},
		{Model: trained, TrainedFrom: weights, Name: "qwen25-05b-order-25", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{
			metric.Quality: read(0.1447, "raw-completion"), metric.Throughput: measured(321.3), metric.FirstOutput: gap("not measured"),
		}},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "other"), Name: "other", Task: recipe.TaskInference, Cells: map[metric.Name]Cell{
			metric.Quality: measured(0.5),
		}},
	}
	againstBase(rows, map[artifact.ID]string{
		weights: `C:\models\Qwen2.5-0.5B\model.safetensors`, base: `C:\models\Qwen2.5-0.5B`, trained: `C:\checkpoints\order-25`,
	})
	row := rows[1]
	if row.Base != "Qwen2.5-0.5B" {
		t.Fatalf("the trained row names its base %q", row.Base)
	}
	for column, want := range map[metric.Name]float64{metric.Quality: 0.1696, metric.Throughput: 505.7} {
		if got := row.Cells[column].AgainstBase; got == nil || *got != want {
			t.Errorf("%s holds its base's reading %v, want %v", column, got, want)
		}
	}
	if row.Cells[metric.FirstOutput].AgainstBase != nil || rows[0].Base != "" || rows[2].Cells[metric.Quality].AgainstBase != nil {
		t.Fatal("a reading was set beside a cell with no reading of its own, or on a row no training made")
	}
	for _, gap := range (Record{Rows: rows}).Gaps(&Record{}) {
		if gap.kind == gapBelowBase || strings.Contains(gap.Title, "beat its base") {
			t.Fatalf("a trained model below its base filed %+v", gap)
		}
	}
}
