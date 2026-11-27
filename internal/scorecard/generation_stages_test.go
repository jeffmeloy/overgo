package scorecard

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/media"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestTheStagesAreTheNewestAttemptsWithItsOwnTotal holds the generation table's parts to the attempt they are of: a row's
// time at a person's size is the median of its newest attempts and its parts are the newest attempt's, so SenseNova's
// 130.7 s stood beside parts of 10.9 s and 120.0 s, which pass it (SQA-512). The parts are said after their attempt's
// own time, their sum.
func TestTheStagesAreTheNewestAttemptsWithItsOwnTotal(t *testing.T) {
	t.Parallel()
	small := media.Preset{Width: 512, Height: 512, Steps: 4, Sampler: "flow Euler"}
	sized := media.Preset{Width: 2048, Height: 2048, Steps: 50, Sampler: "flow Euler"}
	row := Row{Name: "SenseNova", Task: recipe.TaskImageGen,
		Cells: map[metric.Name]Cell{metric.FirstOutput: {Value: 12_300, Preset: small}},
		Warm:  []Cell{{Value: 130_670, Preset: sized, Contract: Contract{Metric: userSizeTimeMetric}}},
		Stages: []processmeasure.Part{
			{Name: "model session leased", Wall: 10_900 * time.Millisecond, Times: 1},
			{Name: "model.routed-image-integrate", Wall: 120_000 * time.Millisecond, Times: 1},
			{Name: "(no part names)", Wall: 1_400 * time.Millisecond, Times: 1},
		}}
	table := (Record{Rows: []Row{row}}).GenerationTable()
	want := "| SenseNova | 130.7 s (2048 x 2048, 50 steps, flow Euler) | 12.3 s (512 x 512, 4 steps, flow Euler) | 132.3 s: model session leased 10.9 s, model.routed-image-integrate 120.0 s |"
	if !strings.Contains(table, want) || !strings.Contains(table, "where its newest attempt at a person's size went") {
		t.Fatalf("the generation table reads\n%s\nwant the parts after their attempt's own time:\n%s", table, want)
	}
}

// TestAPersonSizeReadingStandsWithoutTheTimingRequestsPreset holds the generation table to every reading that names
// what it ran: a row whose timing request recorded no preset had been left out whole, its measured 374 s at 81 frames
// of 832 x 480 hidden with it (SQA-513). Such a row shows its person-size reading with a dash for the timing request.
func TestAPersonSizeReadingStandsWithoutTheTimingRequestsPreset(t *testing.T) {
	t.Parallel()
	sized := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 50, Sampler: "flow UniPC"}
	row := Row{Name: "Wan", Task: recipe.TaskVideoGen, Cells: map[metric.Name]Cell{metric.FirstOutput: {Value: 13_000}},
		Warm: []Cell{{Value: 374_000, Preset: sized, Contract: Contract{Metric: userSizeTimeMetric}}}}
	want := "| Wan | 374.0 s (832 x 480, 81 frames, 50 steps, flow UniPC) | — | — |"
	if table := (Record{Rows: []Row{row}}).GenerationTable(); !strings.Contains(table, want) {
		t.Fatalf("the generation table reads\n%s\nwant the person-size reading beside a dash:\n%s", table, want)
	}
}
