package landing

import (
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
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
	// Measurement, Turn and Device are the waits within the run: for a timed
	// measurement of the cycle to finish, for the host's lane turn, and for
	// the device to admit a lane's steps.
	Measurement time.Duration `json:"measurement_wait,omitzero"`
	Turn        time.Duration `json:"turn_wait,omitzero"`
	Device      time.Duration `json:"device_wait,omitzero"`
	// Stages is the time of each stage that ran, its waits included.
	Stages map[string]time.Duration `json:"stages,omitempty"`
}

// Outcomes of a run that no stage refused.
const (
	outcomeLanded    = "landed"
	outcomePreflight = "preflight"
	outcomeConflict  = "conflict"
)

// spentRecord holds the runs' records in the checkout's process state.
const spentRecord = "landing_waits.json"

// recordSpent closes the run's record and adds it to the checkout's.
func (l *landing) recordSpent() error {
	l.spent.Ended = time.Now()
	record, earlier, err := stateRecords[spent](l.Repo, spentRecord)
	if err != nil {
		return err
	}
	return jsonfile.Write(record, append(earlier, l.spent), clioptions.PrivateFileMode)
}
