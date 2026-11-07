package landing

import (
	"slices"
	"time"
)

// Pace is what a week of the gate's runs in a checkout says of the work's pace: commits a day, the share of commit
// attempts the checks refused, and the hours between one run's end and the next one's start beyond an hour, when no
// commit was being made. The week to Nov 7 read 44 idle hours and 45% refused.
type Pace struct {
	LandingsPerDay float64 `json:"landings_per_day"`
	RefusedShare   float64 `json:"refused_share"`
	IdleHours      float64 `json:"idle_hours"`
}

// The window a pace is read over, the gap between runs past which the work counts as idle, and the day landings are
// counted in.
const (
	paceWindow = 7 * 24 * time.Hour
	idleGap    = time.Hour
	paceDay    = 24 * time.Hour
)

// WeekPace reads the pace of the week before now from the gate's runs the checkout at root records; a preflight is
// no commit attempt and is left out.
func WeekPace(root string, now time.Time) (Pace, error) {
	_, runs, err := stateRecords[spent](root, spentRecord)
	if err != nil {
		return Pace{}, err
	}
	since := now.Add(-paceWindow)
	runs = slices.DeleteFunc(runs, func(run spent) bool { return run.Outcome == outcomePreflight || run.Started.Before(since) })
	slices.SortFunc(runs, func(a, b spent) int { return a.Started.Compare(b.Started) })
	var pace Pace
	landed := 0
	last := since
	for _, run := range runs {
		if run.Outcome == outcomeLanded {
			landed++
		}
		if gap := run.Started.Sub(last); gap > idleGap {
			pace.IdleHours += gap.Hours()
		}
		last = run.Ended
	}
	if gap := now.Sub(last); gap > idleGap {
		pace.IdleHours += gap.Hours()
	}
	pace.LandingsPerDay = float64(landed) / (paceWindow.Hours() / paceDay.Hours())
	if len(runs) != 0 {
		pace.RefusedShare = float64(len(runs)-landed) / float64(len(runs))
	}
	return pace, nil
}
