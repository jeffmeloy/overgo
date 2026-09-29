package runrecord

import (
	"errors"
	"fmt"
	"strings"
)

// laneOutcome is the shared terminal vocabulary for verification lanes.
type laneOutcome string

const (
	lanePassed      laneOutcome = "pass"
	LaneFailed      laneOutcome = "fail"
	LaneUnavailable laneOutcome = "unavailable"
	LaneEmpty       laneOutcome = "empty"
)

type laneError struct {
	outcome laneOutcome
	detail  string
}

func (e laneError) Error() string {
	return fmt.Sprintf("outcome=%s: %s", e.outcome, e.detail)
}

// LaneError creates a non-passing terminal result. Passing lanes return nil.
func LaneError(outcome laneOutcome, detail string) error {
	detail = strings.TrimSpace(detail)
	if outcome == lanePassed || detail == "" {
		return errors.New("run record: invalid lane error")
	}
	switch outcome {
	case LaneFailed, LaneUnavailable, LaneEmpty:
		return laneError{outcome: outcome, detail: detail}
	default:
		return errors.New("run record: invalid lane outcome")
	}
}
