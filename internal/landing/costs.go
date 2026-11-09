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

// Attempt is one worker attempt at a step: the worker that ran it, when it started and, once it ended, its wall; or,
// where the driver's run ended without the attempt's exit (a machine powered off), that it was interrupted, its wall
// unknown, which the driver's next start records.
type Attempt struct {
	Step        string        `json:"step"`
	Worker      string        `json:"worker"`
	Started     time.Time     `json:"started"`
	Wall        time.Duration `json:"wall,omitzero"`
	Interrupted bool          `json:"interrupted,omitzero"`
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

// cost is what the landed steps cost: the worker attempts at them, how many of those were interrupted, the workers
// that ran them, their wall, an attempt still running counted to now and an interrupted one not at all, and the gate's
// own wall to now.
type cost struct {
	attempts, interrupted int
	workers               []string
	workerWall, gate      time.Duration
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
	// An attempt is its start and, once it ended, its end: the end's wall stands, a start alone runs to now, and one the
	// driver's next start found interrupted is counted with no wall, since how long it ran before the run ended is
	// unknown: charging it to now would count the downtime (SQA-356).
	walls, started, interrupted, workers := map[string]time.Duration{}, map[string]time.Time{}, map[string]bool{}, map[string]string{}
	var order []string
	for _, attempt := range recorded {
		if !slices.Contains(steps, attempt.Step) {
			continue
		}
		key := attempt.Step + " " + attempt.Started.String()
		if _, seen := walls[key]; !seen {
			order = append(order, key)
			walls[key], started[key], workers[key] = now.Sub(attempt.Started), attempt.Started, attempt.Worker
		}
		if attempt.Wall > 0 {
			walls[key] = attempt.Wall
		}
		if attempt.Interrupted {
			walls[key], interrupted[key] = 0, true
		}
	}
	// The workers in the order their attempts started, each moved last when it ran again, so the last named ran the
	// attempt that lands.
	slices.SortStableFunc(order, func(a, b string) int { return started[a].Compare(started[b]) })
	for _, key := range order {
		// An interrupted attempt whose start was not recorded names no worker.
		if workers[key] != "" {
			spent.workers = append(slices.DeleteFunc(spent.workers, func(worker string) bool { return worker == workers[key] }), workers[key])
		}
		spent.workerWall += walls[key]
		if interrupted[key] {
			spent.interrupted++
		}
	}
	spent.attempts = len(order)
	return spent, nil
}

// said is the cost as the message's Cost line says it, and its trailers.
func (spent cost) said() (line string, trailers []string) {
	gate := spent.gate.Round(time.Second)
	switch {
	case spent.attempts == 0:
		// No record is no proof of an interactive session: a driver whose record was not written leaves none either.
		line = fmt.Sprintf("Cost: no worker attempt recorded; the gate took %s.", gate)
	case spent.interrupted > 0:
		line = fmt.Sprintf("Cost: %d worker attempts by %s, %d before the one that landed, %d interrupted with their wall unknown, in %s of the rest's wall; the gate took %s.",
			spent.attempts, strings.Join(spent.workers, ", "), spent.attempts-1, spent.interrupted, spent.workerWall.Round(time.Second), gate)
	default:
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
