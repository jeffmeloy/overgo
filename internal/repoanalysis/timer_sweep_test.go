package repoanalysis

import (
	"path/filepath"
	"strings"
	"testing"
)

// liveFixedTimerCensus takes the census over the repository's own tree.
func liveFixedTimerCensus(t *testing.T) []FixedTimer {
	t.Helper()
	repository, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	live, err := DiscoverGo(repository, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	census, err := FixedTimerCensus(live)
	if err != nil {
		t.Fatal(err)
	}
	return census
}

// TestFixedTimersRetiredFromProduction holds the production tree free of
// fixed wall-clock bounds: every wait in a non-test file observes the
// awaited operation and ends with it, with its failure, or with the
// caller's cancellation.
func TestFixedTimersRetiredFromProduction(t *testing.T) {
	t.Parallel()
	for _, timer := range liveFixedTimerCensus(t) {
		if !strings.HasSuffix(timer.File, "_test.go") {
			t.Errorf("production fixed timer: %s %s x%d", timer.File, timer.Function, timer.Count)
		}
	}
}

// TestFixedTimersRetired holds the whole tree, tests included, free of
// fixed wall-clock bounds: a test waits on the signal of the operation it
// exercises, and the test binary's declared budget reports a hang.
func TestFixedTimersRetired(t *testing.T) {
	t.Parallel()
	for _, timer := range liveFixedTimerCensus(t) {
		t.Errorf("fixed timer: %s %s x%d", timer.File, timer.Function, timer.Count)
	}
}
