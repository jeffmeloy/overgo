package runrecord

import (
	"errors"
	"fmt"
	"strings"
)

// LaneOutcome is the shared terminal vocabulary for verification lanes.
type LaneOutcome string

const (
	LanePassed      LaneOutcome = "pass"
	LaneFailed      LaneOutcome = "fail"
	LaneUnavailable LaneOutcome = "unavailable"
	LaneEmpty       LaneOutcome = "empty"
)

type laneError struct {
	outcome LaneOutcome
	detail  string
}

func (e laneError) Error() string {
	return fmt.Sprintf("outcome=%s: %s", e.outcome, e.detail)
}

// LaneError creates a non-passing terminal result. Passing lanes return nil.
func LaneError(outcome LaneOutcome, detail string) error {
	detail = strings.TrimSpace(detail)
	if outcome == LanePassed || detail == "" {
		return errors.New("run record: invalid lane error")
	}
	switch outcome {
	case LaneFailed, LaneUnavailable, LaneEmpty:
		return laneError{outcome: outcome, detail: detail}
	default:
		return errors.New("run record: invalid lane outcome")
	}
}

// LaneOutcomeOf maps an error to its terminal lane outcome.
func LaneOutcomeOf(err error) LaneOutcome {
	if err == nil {
		return LanePassed
	}
	if terminal, ok := errors.AsType[laneError](err); ok {
		return terminal.outcome
	}
	return LaneFailed
}
