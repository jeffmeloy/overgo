package scorecard

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
)

// TestTheSpeedTableRatesAPromptByTheTokensItRead holds the README's prompt processing to the work a prefill did
// (SQA-336): the table divided the length bucket by the prefill time, so a request that read 3,072 tokens uncached in a
// second, in the 4,096 bucket, printed 4,096 tok/s. It prints 3,072 tok/s; a reading whose record holds no token
// counts prints none.
func TestTheSpeedTableRatesAPromptByTheTokensItRead(t *testing.T) {
	t.Parallel()
	bucket := evaluation.PromptLengths[0]
	read := func(input, cached uint64) []Cell {
		var observation runrecord.ServingObservation
		observation.Hardware = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(time.Second)}}
		observation.Usage = runrecord.ServingUsage{InputTokens: input, CachedTokens: cached}
		var summary servingSummary
		summary.readLength(bucket, observation)
		return lengthCells(recipe.TaskInference, &summary)
	}
	rate := map[metric.Name]Cell{metric.Throughput: {Value: 50, Unit: "tok/s", Contract: Contract{Metric: warmRateMetric}}}
	record := Record{Rows: []Row{
		{Name: "read", Task: recipe.TaskInference, Cells: rate, Lengths: read(3200, 128)},
		{Name: "uncounted", Task: recipe.TaskInference, Cells: rate, Lengths: read(0, 0)},
	}}
	table := record.SpeedTable()
	if !strings.Contains(table, "| read | 3,072 tok/s (1.00 s) |") || !strings.Contains(table, "| uncounted | — |") {
		t.Fatalf("the speed table =\n%s\nwant 3,072 tok/s for the tokens read uncached, and none for a reading without counts", table)
	}
}
