package scorecard

import (
	"bytes"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestACarriedReadingOutsideTheSelectionIsNoGap holds the scorecard's gaps to the readings a run owes
// (carried-unselected, SQA-527): a published record that marks a row's model as one the run's list leaves out
// (Row.Unchosen), the record Propose files gaps from, files none of its carried readings, which stand dated until a run
// is given the model, while a listed model's carried reading and an unchosen model's reading never taken stay gaps.
// Qwen3.5-9B's six carried readings were filed as unmeasured at 3a7b3877d while the run measured Qwen3.5-4B, the model
// its list names for the path the two share.
func TestACarriedReadingOutsideTheSelectionIsNoGap(t *testing.T) {
	t.Parallel()
	previous := time.Date(2026, 11, 28, 12, 14, 2, 0, time.UTC)
	row := func(name string) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = Cell{NotApplicable: true, Reason: "not read in this fixture"}
		}
		cells[metric.Throughput] = Cell{Value: 40, Unit: "tok/s", Source: testutil.ArtifactID(t, artifact.KindEvidence, name), Newest: previous.Add(-time.Hour).UnixNano()}
		cells[metric.Load] = gap("never measured")
		return Row{Model: testutil.ArtifactID(t, artifact.KindModel, name), Name: name, Task: recipe.TaskInference, Cells: cells}
	}
	directory := t.TempDir()
	carried := Record{Commit: "b", Rows: []Row{row("Qwen3.5-9B"), row("Qwen3.5-4B")}}.Carry(previous)
	path := testcheck.Must(Publish(directory, previous.Add(time.Hour), carried))(t)
	// The run's publication marks the row of the model its list leaves out, in the record file Propose reads.
	decoder := json.NewDecoder(bytes.NewReader(testcheck.Must(os.ReadFile(path))(t)))
	decoder.UseNumber()
	var document map[string]any
	testcheck.NoError(t, decoder.Decode(&document))
	document["rows"].([]any)[0].(map[string]any)["unchosen"] = true
	testcheck.NoError(t, os.WriteFile(path, testcheck.Must(json.Marshal(document))(t), 0o600))

	records := testcheck.Must(Records(directory))(t)
	if len(records) != 1 {
		t.Fatalf("read %d records, want the one published", len(records))
	}
	gaps := records[0].Gaps(nil)
	at := slices.IndexFunc(gaps, func(gap Gap) bool { return gap.Key == gapKey(gapUnmeasured, recipe.TaskInference) })
	if at < 0 {
		t.Fatalf("the record files no unmeasured inference gap: %+v", gaps)
	}
	detail := gaps[at].Detail
	if strings.Contains(detail, "Qwen3.5-9B "+string(metric.Throughput)) {
		t.Errorf("a carried reading of a model the run's list leaves out is filed as owed:\n%s", detail)
	}
	for _, owed := range []string{"Qwen3.5-4B " + string(metric.Throughput), "Qwen3.5-9B " + string(metric.Load)} {
		if !strings.Contains(detail, owed) {
			t.Errorf("the gap lacks %q, a listed model's carried reading or a reading never taken, which stay owed:\n%s", owed, detail)
		}
	}
}
