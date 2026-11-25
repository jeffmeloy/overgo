package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
)

// TestTheServedTablesPeakIsTheMostAMeasuredAttemptTook holds the served-model table's peak device memory to the most a
// measured serving attempt took: Wan's 32 x 16 timing request peaked at 12.2 GiB on the card and its 81 frames at
// 832 x 480 at 18.1 GiB in the closing run of 2026-11-25, and the table showed the 12.2 (SQA-510). A row's request at a
// person's size counts beside its timing request.
func TestTheServedTablesPeakIsTheMostAMeasuredAttemptTook(t *testing.T) {
	t.Parallel()
	wan := Row{Name: "Wan", Task: recipe.TaskVideoGen,
		Cells: map[metric.Name]Cell{metric.PeakMemory: {Value: 12.2 * gibibyte, Unit: "bytes"}, metric.Quality: {Reason: "a fixture measures nothing"}},
		Warm:  []Cell{{Value: 18.1 * gibibyte, Unit: "bytes", Contract: Contract{Metric: userSizeMemoryMetric}}}}
	words := ServedWords{Models: []ServedModel{{Row: "Wan", Shown: "Wan2.1-T2V 1.3B", Modality: "text-to-video", InputOutput: "text → video"}}}
	table := testcheck.Must((Record{Rows: []Row{wan}}).ServedTable(words))(t)
	if !strings.Contains(table, "| Wan2.1-T2V 1.3B | text-to-video | text → video | — | 18.1 GiB |") {
		t.Fatalf("the served-model table reads\n%s\nwant Wan's peak at its person's size, 18.1 GiB", table)
	}
}
