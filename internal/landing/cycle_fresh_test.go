package landing

import (
	"slices"
	"testing"
)

// TestEveryCycleLaneThatRunsGoTestRunsAfresh holds the cycle's real-model evidence to the bytes it reads (SQA-315):
// go test keys its cache on inputs inside the module, so the models and needle-reference lanes, run without -count=1,
// replayed a pass over a model or fixture that changed outside it. Every cycle lane whose command runs go test carries
// -count=1, a lane added later among them, and one without it is named.
func TestEveryCycleLaneThatRunsGoTestRunsAfresh(t *testing.T) {
	t.Parallel()
	tested := 0
	for _, lane := range cycleLanes {
		arguments := lane.command("")
		if len(arguments) == 0 || arguments[0] != "test" {
			continue
		}
		tested++
		if !slices.Contains(arguments, "-count=1") {
			t.Errorf("the cycle lane %s runs go test %v without -count=1, so its cache replays a pass over inputs outside the module", lane.name, arguments[1:])
		}
	}
	if tested == 0 {
		t.Fatal("no cycle lane runs go test, so this shows nothing")
	}
}
