package seriesforecast

import (
	"sort"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestForecastWallDecomposition attributes the head-to-head gap: load wall
// and per-forecast wall, medians over repeats, printed for the performance
// leg's evidence. Not an assertion — the at-or-below verdict lives in the
// plan measurement, and this probe feeds it.
func TestForecastWallDecomposition(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": loads ~930MB weights")
	}
	g := readGolden(t)
	dir := artifactDir(t)

	loadStart := time.Now()
	model, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	loadWall := time.Since(loadStart)

	const repeats = 9
	walls := make([]time.Duration, 0, repeats)
	for r := 0; r < repeats; r++ {
		start := time.Now()
		for _, c := range g.Cases {
			if _, err := model.Forecast(c.Context); err != nil {
				t.Fatal(err)
			}
		}
		walls = append(walls, time.Since(start))
	}
	sort.Slice(walls, func(i, j int) bool { return walls[i] < walls[j] })
	t.Logf("load=%v forecast(both cases) median=%v min=%v max=%v", loadWall, walls[repeats/2], walls[0], walls[repeats-1])
}
