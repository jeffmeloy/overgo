package landing

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
)

// attemptsRecord is the record in a checkout's process state of each worker attempt the driver ran at a step: one entry
// as the attempt starts, and one with its wall as it ends.
const attemptsRecord = "worker_attempts.json"

// Attempt is one worker attempt at a step: the worker that ran it, when it started and, once it ended, its wall.
type Attempt struct {
	Step    string        `json:"step"`
	Worker  string        `json:"worker"`
	Started time.Time     `json:"started"`
	Wall    time.Duration `json:"wall,omitzero"`
}

// RecordAttempt adds the attempt to the record in the process state of the checkout at root.
func RecordAttempt(root string, attempt Attempt) error {
	record, earlier, err := stateRecords[Attempt](root, attemptsRecord)
	if err != nil {
		return err
	}
	return jsonfile.Write(record, append(earlier, attempt), clioptions.PrivateFileMode)
}

// The trailers a landing's cost is said in, which git log reads back (go run ./cmd/loop -costs).
const (
	// WorkerAttemptsTrailer names how many worker attempts the driver recorded at the landed steps.
	WorkerAttemptsTrailer = "Worker-Attempts"
	// WorkerWallTrailer names the seconds of those attempts' wall.
	WorkerWallTrailer = "Worker-Wall-Seconds"
	// GateWallTrailer names the seconds of the gate's run that landed them.
	GateWallTrailer = "Gate-Wall-Seconds"
	// WorkerTrailer names each worker that ran an attempt, the last the one that landed.
	WorkerTrailer = "Worker"
)

// cost is what the landed steps cost: the worker attempts at them, the workers that ran them, their wall, an attempt
// still running counted to now, and the gate's own wall to now.
type cost struct {
	attempts         int
	workers          []string
	workerWall, gate time.Duration
}

// cost reads, at now, what the landed steps cost: the attempts at them the driver recorded, the one that lands them
// still running, and the gate's run.
func (l *landing) cost(now time.Time) (cost, error) {
	_, recorded, err := stateRecords[Attempt](l.Repo, attemptsRecord)
	if err != nil {
		return cost{}, err
	}
	named, err := rows(l.Ref)
	if err != nil {
		return cost{}, err
	}
	steps := make([]string, len(named))
	for index, row := range named {
		steps[index] = row.String()
	}
	spent := cost{gate: now.Sub(l.spent.Started)}
	// An attempt is its start and, once it ended, its end: the end's wall stands, a start alone runs to now.
	walls := map[string]time.Duration{}
	var order []string
	for _, attempt := range recorded {
		if !slices.Contains(steps, attempt.Step) {
			continue
		}
		key := attempt.Step + " " + attempt.Started.String()
		if _, seen := walls[key]; !seen {
			order = append(order, key)
			walls[key] = now.Sub(attempt.Started)
			if !slices.Contains(spent.workers, attempt.Worker) {
				spent.workers = append(spent.workers, attempt.Worker)
			}
		}
		if attempt.Wall > 0 {
			walls[key] = attempt.Wall
		}
	}
	spent.attempts = len(order)
	for _, key := range order {
		spent.workerWall += walls[key]
	}
	return spent, nil
}

// said is the cost as the message's Cost line says it, and its trailers.
func (spent cost) said() (line string, trailers []string) {
	gate := spent.gate.Round(time.Second)
	if spent.attempts == 0 {
		line = fmt.Sprintf("Cost: no worker attempt recorded, the work an interactive session's; the gate took %s.", gate)
	} else {
		line = fmt.Sprintf("Cost: %d worker attempts by %s, %d before the one that landed, in %s of their wall; the gate took %s.",
			spent.attempts, strings.Join(spent.workers, ", "), spent.attempts-1, spent.workerWall.Round(time.Second), gate)
	}
	trailers = []string{
		fmt.Sprintf("%s: %d", WorkerAttemptsTrailer, spent.attempts),
		fmt.Sprintf("%s: %d", WorkerWallTrailer, int64(spent.workerWall.Seconds())),
		fmt.Sprintf("%s: %d", GateWallTrailer, int64(spent.gate.Seconds())),
	}
	for _, worker := range spent.workers {
		trailers = append(trailers, WorkerTrailer+": "+worker)
	}
	return line, trailers
}
