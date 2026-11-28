package scorecard

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAReferenceReadsAMeasureOnce holds a reference row to one reading of each measure, its newest (reference-newest): a
// model whose warm request's throughput the model it replaced read in its list of warm requests, and which it reads in
// its own headline cell, is judged against its own readings alone. MiniCPM5-1B's and Qwen2.5-0.5B's throughput, read at
// about 355 and 534 tok/s since 2026-11-07, were filed at 3a7b3877d as 96% and 95% declines from the 8765 and 9838
// tok/s the models they replaced read on 2026-10-31, readings of requests that ended at their first token (rateTokens).
func TestAReferenceReadsAMeasureOnce(t *testing.T) {
	t.Parallel()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "environment")
	replaced, model := testutil.ArtifactID(t, artifact.KindModel, "native-chat"), testutil.ArtifactID(t, artifact.KindModel, "MiniCPM5-1B")
	cold := Contract{Metric: "output tokens over decode time", Origin: originServing}
	warm := Contract{Metric: warmRateMetric, Origin: originServing}
	reading := func(source string, value float64, contract Contract) Cell {
		return Cell{Value: value, Unit: "tok/s", Source: testutil.ArtifactID(t, artifact.KindEvidence, source), Environment: environment, Contract: contract}
	}
	records := []Record{{Commit: "replaced", Rows: []Row{{
		Model: replaced, Name: "MiniCPM5-1B-native-chat", Task: recipe.TaskInference,
		Cells: map[metric.Name]Cell{metric.Throughput: reading("cold", 369, cold)}, Warm: []Cell{reading("first token alone", 8765, warm)},
	}}}}
	for index, value := range []float64{354, 351, 350, 352, 353, 355} {
		records = append(records, Record{Commit: fmt.Sprint("own ", index), Rows: []Row{{
			Model: model, Name: "MiniCPM5-1B", Task: recipe.TaskInference, Succeeds: []artifact.ID{replaced},
			Cells: map[metric.Name]Cell{metric.Throughput: reading(fmt.Sprint("own ", index), value, warm)},
		}}})
	}
	reference := Reference(records)
	if reference == nil {
		t.Fatal("six records after the first have no reference")
	}
	for _, pair := range records[len(records)-1].Pairs(*reference) {
		if pair.Before == nil {
			continue
		}
		if read := pair.Before.Measured(metric.Throughput); len(read) != 1 || read[0].Contract != warm {
			t.Errorf("the reference reads the warm request's throughput %d times, want once, its own newest: %+v", len(read), read)
		}
		for _, comparison := range pair.Compare(metric.Throughput) {
			if comparison.Verdict == regressed {
				t.Errorf("355 tok/s after readings of 350 to 354 regressed: %s", comparison.Why)
			}
		}
	}
}
