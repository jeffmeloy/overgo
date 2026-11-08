package scorecard

import (
	"testing"
	"time"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestTheSpeedTableReadsWarmCellsAlone holds the README's speed table to a record's warm cells: a row read warm at both
// longer prompts and on its warm request gives its prompt processing at each, the prompt's tokens over its first
// output's seconds, and its token generation; a row whose cells are its first request's and the cold longer-prompt
// readings a record held before speed was read warm gives no line; a row with only a warm rate gives dashes beside it;
// an image row and a stale row give none.
func TestTheSpeedTableReadsWarmCellsAlone(t *testing.T) {
	t.Parallel()
	short, long := evaluation.PromptLengths[0], evaluation.PromptLengths[1]
	length := func(prompt int, ms float64, measure string) Cell {
		return Cell{Value: ms, Unit: "ms", Prompt: prompt, Rate: float64(prompt) / time.Duration(ms*float64(time.Millisecond)).Seconds(), Contract: Contract{Metric: measure}}
	}
	rate := func(value float64, measure string) map[metric.Name]Cell {
		return map[metric.Name]Cell{metric.Throughput: {Value: value, Unit: "tok/s", Contract: Contract{Metric: measure}}}
	}
	record := Record{Rows: []Row{
		{Name: "warm", Task: recipe.TaskInference, Cells: rate(134, warmRateMetric),
			Lengths: []Cell{length(short, 1610, lengthFirstMetric(short)), length(long, 5260, lengthFirstMetric(long))}},
		{Name: "cold", Task: recipe.TaskInference, Cells: rate(67, "output tokens over decode time"),
			Lengths: []Cell{length(short, 2840, "time to prefill, a fresh prompt of 4096 tokens")}},
		{Name: "rate only", Task: recipe.TaskInference, Cells: rate(97, warmRateMetric)},
		{Name: "painter", Task: recipe.TaskImageGen, Cells: rate(5, warmRateMetric)},
		{Name: "stale", Task: recipe.TaskInference, Cells: rate(50, warmRateMetric), Stale: "its activation changed"},
	}}
	want := "| Model | pp, 4,096-token prompt | pp, 16,384-token prompt | tg |\n| --- | --- | --- | --- |\n" +
		"| warm | 2,544 tok/s (1.61 s) | 3,115 tok/s (5.26 s) | 134 tok/s |\n" +
		"| rate only | — | — | 97 tok/s |\n"
	if got := record.SpeedTable(); got != want {
		t.Fatalf("the speed table =\n%s\nwant\n%s", got, want)
	}
	for count, written := range map[int]string{0: "0", 999: "999", 1000: "1,000", 16384: "16,384", 1234567: "1,234,567"} {
		if got := thousands(count); got != written {
			t.Errorf("thousands(%d) = %q, want %q", count, got, written)
		}
	}
}
