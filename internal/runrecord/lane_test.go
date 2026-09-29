package runrecord

import (
	"strings"
	"testing"
)

func TestLaneOutcomeContract(t *testing.T) {
	for _, outcome := range []laneOutcome{LaneFailed, LaneUnavailable, LaneEmpty} {
		err := LaneError(outcome, "fixture")
		if !strings.Contains(err.Error(), "outcome="+string(outcome)) {
			t.Fatalf("%s error = %q", outcome, err)
		}
	}
	if err := LaneError(lanePassed, "impossible"); err == nil || strings.Contains(err.Error(), "outcome=") {
		t.Fatalf("passing error accepted: %v", err)
	}
}
