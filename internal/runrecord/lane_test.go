package runrecord

import (
	"errors"
	"strings"
	"testing"
)

func TestLaneOutcomeContract(t *testing.T) {
	if got := LaneOutcomeOf(nil); got != LanePassed {
		t.Fatalf("nil outcome = %q", got)
	}
	for _, outcome := range []LaneOutcome{LaneFailed, LaneUnavailable, LaneEmpty} {
		err := LaneError(outcome, "fixture")
		if got := LaneOutcomeOf(err); got != outcome {
			t.Fatalf("%s outcome = %q", outcome, got)
		}
		if !strings.Contains(err.Error(), "outcome="+string(outcome)) {
			t.Fatalf("%s error = %q", outcome, err)
		}
	}
	if got := LaneOutcomeOf(errors.New("plain")); got != LaneFailed {
		t.Fatalf("plain error outcome = %q", got)
	}
	if err := LaneError(LanePassed, "impossible"); err == nil || LaneOutcomeOf(err) != LaneFailed {
		t.Fatalf("passing error accepted: %v", err)
	}
}
