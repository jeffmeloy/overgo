package landing

import (
	"cmp"
	"context"
	"fmt"
	"time"

	"overgo/internal/plan"
)

const (
	// spaceClass names the stage that reads free space, and the class of its refusal.
	spaceClass = "space"
	// gibibyte is the unit free space is said in.
	gibibyte = 1 << 30
)

// spaceNeeds is what a run of a commit's checks and a measurement run were recorded to need free: the greatest fall of
// the volume's free space during a landing that landed in the week, and during the last measurement run, each watched
// as it ran (WatchFall). A measurement run writes beside commits, so the floor is their sum. None recorded is zero, and
// says so.
type spaceNeeds struct {
	landing, measurement uint64
}

// floor is the least free space a commit's checks may start with.
func (n spaceNeeds) floor() uint64 { return n.landing + n.measurement }

// recordedSpaceNeeds reads the falls the checkout at root, and the trunk's checkout trunk, recorded: on 2026-11-15
// dispatch-next/do was refused at 34.1 GiB free below a floor of two figures measured by hand a month before, while a
// runaway's page file had taken the disk, 268 GB free fifteen minutes later (space-record/do).
func recordedSpaceNeeds(root, trunk string, now time.Time) (spaceNeeds, error) {
	landing, err := WeekLandingFall(root, now)
	if err != nil {
		return spaceNeeds{}, err
	}
	needs := spaceNeeds{landing: landing}
	if run, read := plan.ReadMeasurementRun(cmp.Or(trunk, root)); read {
		needs.measurement = run.Fall
	}
	return needs, nil
}

// WeekLandingFall is the greatest fall of free space a landing that landed in the checkout at root recorded in the
// week before now, the window the gate's pace is read over; zero where none did. The last one alone let a change to the
// plan, which lands falling almost nothing, set the floor near zero for the landing after it (2026-11-16: 0.0 GiB),
// and the greatest ever would hold one run's fall for good, a runaway's page file among it (space-week/do).
func WeekLandingFall(root string, now time.Time) (uint64, error) {
	_, runs, err := stateRecords[spent](root, spentRecord)
	if err != nil {
		return 0, err
	}
	var greatest uint64
	for _, run := range settled(runs) {
		if run.Outcome == outcomeLanded && !run.Started.Before(now.Add(-paceWindow)) {
			greatest = max(greatest, run.Fall)
		}
	}
	return greatest, nil
}

// space reads, once and before any other stage, the free space of the volume that holds the commit's temporary
// directory, and refuses the run below the floor the recorded falls set, or when the read fails: a volume that fills
// fails every later stage for a reason no stage names. A read that failed is said as that, with no figure: the
// reader's zero is not a reading.
func (l *landing) space(context.Context) error {
	needs, err := recordedSpaceNeeds(l.Repo, l.trunk, time.Now())
	if err != nil {
		return err
	}
	free, err := l.Free(l.scratch)
	if err != nil {
		return fmt.Errorf("the free space of the volume of %s could not be read, so nothing says it is above the floor of %d bytes: %w", l.scratch, needs.floor(), err)
	}
	found := fmt.Sprintf("free %.1f GiB, floor %.1f GiB: a landing in the week fell at most %.1f GiB, the last measurement run %.1f GiB, none recorded reading 0",
		float64(free)/gibibyte, float64(needs.floor())/gibibyte, float64(needs.landing)/gibibyte, float64(needs.measurement)/gibibyte)
	if free < needs.floor() {
		return fmt.Errorf("%s (%d below %d bytes) on the volume of %s: that is below the floor; go run ./cmd/loop -leftovers lists what can be removed",
			found, free, needs.floor(), l.scratch)
	}
	l.details[spaceClass] = found
	return nil
}
