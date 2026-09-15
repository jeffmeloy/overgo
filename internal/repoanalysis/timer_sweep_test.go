package repoanalysis

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestFixedTimersRetiredFromProduction holds the production tree free of
// fixed wall-clock bounds: every wait in a non-test file observes the
// awaited operation and ends with it, with its failure, or with the
// caller's cancellation.
func TestFixedTimersRetiredFromProduction(t *testing.T) {
	t.Parallel()
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
	for _, timer := range census {
		if !strings.HasSuffix(timer.File, "_test.go") {
			t.Errorf("production fixed timer: %s %s x%d", timer.File, timer.Function, timer.Count)
		}
	}
}
