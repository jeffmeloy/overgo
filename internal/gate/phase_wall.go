package gate

import (
	"cmp"
	"fmt"
	"slices"
	"time"

	"overgo/internal/runrecord"
)

// PhaseWall is one row of the per-phase wall table: a recorded gate step
// with its wall and outcome.
type PhaseWall struct {
	Name    string
	Phase   runrecord.Phase
	Outcome runrecord.StepOutcome
	Wall    time.Duration
}

// PhaseWallTable orders the recorded steps by wall, longest first, ties by
// name; a step without wall sorts last.
func PhaseWallTable(steps []runrecord.GateStep) []PhaseWall {
	rows := make([]PhaseWall, 0, len(steps))
	for _, step := range steps {
		rows = append(rows, PhaseWall{Name: step.Name, Phase: step.Phase, Outcome: step.Outcome, Wall: time.Duration(step.DurationNS)})
	}
	slices.SortStableFunc(rows, func(left, right PhaseWall) int {
		return cmp.Or(cmp.Compare(right.Wall, left.Wall), cmp.Compare(left.Name, right.Name))
	})
	return rows
}

// phaseWallMark: the outcome shown beside a row; an executed success shows
// nothing.
func phaseWallMark(outcome runrecord.StepOutcome) string {
	if outcome == runrecord.StepSucceeded {
		return ""
	}
	return " " + string(outcome)
}

// FormatPhaseWallTable renders the table under a heading: one line per step
// with its wall, name, phase and non-success outcome.
func FormatPhaseWallTable(rows []PhaseWall, total time.Duration) []string {
	lines := make([]string, 0, len(rows)+1)
	lines = append(lines, fmt.Sprintf("phase wall: total=%s steps=%d", total.Round(time.Millisecond), len(rows)))
	for _, row := range rows {
		lines = append(lines, fmt.Sprintf("  %-9s %-12s %s%s", row.Wall.Round(time.Millisecond), row.Name, row.Phase, phaseWallMark(row.Outcome)))
	}
	return lines
}
