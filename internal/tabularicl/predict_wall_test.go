package tabularicl

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestPredictWall: matched-protocol wall measurement against the reference
// server's /api/predict "seconds" field. The reference handler compiles and
// loads the resident runtime BEFORE its timer starts, so its seconds is warm
// predict-only; the match is: load each head once, run every golden case
// predictWallRuns times, report the median. Numbers land in docs/plan.json
// (rung5-tabfm performance-leg); this test keeps the protocol reproducible.
const predictWallRuns = 5

func TestPredictWall(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": loads ~6.5GB weights per head")
	}
	golden := readGolden(t)
	for _, task := range Tasks() {
		loadStart := time.Now()
		head, err := LoadHead(headDir(t, task))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: load %.2fs", task, time.Since(loadStart).Seconds())
		for _, c := range golden.Cases {
			if c.Task != task {
				continue
			}
			walls := make([]float64, predictWallRuns)
			for run := range walls {
				start := time.Now()
				if _, err := head.Predict(c.X, c.Y, c.Rows, c.Cols, c.TrainRows, nil); err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				walls[run] = time.Since(start).Seconds()
			}
			slices.Sort(walls)
			t.Logf("%s: median %.4fs over %d runs (min %.4f max %.4f)",
				c.Name, walls[len(walls)/2], predictWallRuns, walls[0], walls[len(walls)-1])
		}
	}
}
