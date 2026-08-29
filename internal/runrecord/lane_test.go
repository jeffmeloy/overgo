package runrecord

import (
	"errors"
	"fmt"
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

func TestJoinedErrorIdentity(t *testing.T) {
	secondary := errors.New("secondary receipt failure")
	err := errors.Join(
		fmt.Errorf("terminal receipt: %w", LaneError(LaneUnavailable, "device")),
		secondary,
	)
	if got := LaneOutcomeOf(err); got != LaneUnavailable {
		t.Fatalf("joined lane outcome = %q", got)
	}
	if !errors.Is(err, secondary) {
		t.Fatal("joined error lost secondary identity")
	}
}
