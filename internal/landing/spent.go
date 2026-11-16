package landing

import (
	"fmt"
	"slices"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
)

// spent is where the time of one run of the gate went, a landing or a
// preflight, landed or refused. The checkout's process state keeps one for
// each run, so a week of them says what limits the pace: the checks
// themselves, a wait for a measurement, a turn or the device, or runs lost to
// a tree that moved. From Started to Ended the tree may not be edited; from
// one run's end to the next one's start the session had it to edit.
type spent struct {
	Reference string    `json:"reference"`
	Started   time.Time `json:"started"`
	Ended     time.Time `json:"ended"`
	// Outcome is landed, preflight, the stage that refused the run, or
	// conflict for a refusal at the commit, where the tree or HEAD moved
	// under the landing.
	Outcome string `json:"outcome"`
	// Queue, Measurement, Turn and Device are the waits within the run: for
	// the commit running in another worktree of the repository to end, for a
	// timed measurement of the cycle to finish, for the host's lane turn, and
	// for the device to admit a lane's steps.
	Queue       time.Duration `json:"queue_wait,omitzero"`
	Measurement time.Duration `json:"measurement_wait,omitzero"`
	Turn        time.Duration `json:"turn_wait,omitzero"`
	Device      time.Duration `json:"device_wait,omitzero"`
	// Stages is the time of each stage that ran, its waits included.
	Stages map[string]time.Duration `json:"stages,omitempty"`
	// Kind is whether the run was a commit attempt or a preflight, empty in a record from before runs said so: a refused
	// preflight and a refused commit share an outcome, the stage that refused them (review-fixes/status-measures).
	Kind string `json:"kind,omitempty"`
	// Process and Began are the run's process, by its id and its start (processcontrol.Running), and Heard the last
	// time the run wrote its record: it is written as the run starts and as each stage starts, and closed as it ends,
	// so a run ended from outside, which wrote no end, reads as ended when last heard, at the stage it had reached
	// (landing-record/do).
	Process int       `json:"process,omitempty"`
	Began   int64     `json:"process_start,omitempty"`
	Heard   time.Time `json:"heard,omitzero"`
	// Fall is the greatest fall of free space the run saw, so far where it has not ended (WatchFall): what the next
	// run's space floor reads, where it landed.
	Fall uint64 `json:"space_fall,omitzero"`
}

// The kinds of a run.
const (
	kindCommit    = "commit"
	kindPreflight = "preflight"
)

// runKind is the kind of a run that is a preflight or not.
func runKind(preflight bool) string {
	if preflight {
		return kindPreflight
	}
	return kindCommit
}

// Outcomes of a run that no stage refused.
const (
	outcomeLanded    = "landed"
	outcomePreflight = "preflight"
	outcomeConflict  = "conflict"
)

// spentRecord holds the runs' records in the checkout's process state.
const spentRecord = "landing_waits.json"

// writeSpent writes the run's record as it stands into the checkout's, in place of what the run wrote before. A record
// that cannot be written is said and changes nothing: the landing stands.
func (l *landing) writeSpent() {
	l.spent.Heard = time.Now()
	if l.fall != nil {
		l.spent.Fall = l.fall()
	}
	record, earlier, err := stateRecords[spent](l.Repo, spentRecord)
	if err == nil {
		earlier = slices.DeleteFunc(earlier, func(run spent) bool { return run.Process == l.spent.Process && run.Started.Equal(l.spent.Started) })
		err = jsonfile.Write(record, append(earlier, l.spent), clioptions.PrivateFileMode)
	}
	if err != nil {
		fmt.Fprintf(l.Log, "land: where the run's time went was not recorded: %v\n", err)
	}
}

// settled are the runs as a reader counts them: one that has not ended while its process runs has no outcome yet and
// is left out, and one whose process has gone without an end was ended from outside, when it was last heard, a failed
// attempt at the stage it had reached. On 2026-11-15 a landing ended from outside at 18:01 left no record, and the
// day's first-attempt rate and gate hours read better than the day was.
func settled(runs []spent) []spent {
	kept := runs[:0]
	for _, run := range runs {
		if run.Ended.IsZero() {
			if processcontrol.Running(run.Process, run.Began) {
				continue
			}
			run.Ended = run.Heard
		}
		kept = append(kept, run)
	}
	return kept
}
