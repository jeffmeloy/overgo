package plan

import (
	"path/filepath"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// LaneRun is one end of a cycle lane in a run of validate -run: the commit the lane ran at, whether it passed, and
// when it ended.
type LaneRun struct {
	Commit string    `json:"commit"`
	Passed bool      `json:"passed"`
	Ended  time.Time `json:"ended"`
}

// LaneRecord is a lane's latest run and its latest pass, which may be the same run, and, for a lane run package by
// package, each package's latest pass (RecordLanePackages).
type LaneRecord struct {
	Last     LaneRun            `json:"last"`
	Passed   LaneRun            `json:"passed,omitzero"`
	Packages map[string]LaneRun `json:"packages,omitempty"`
}

// laneRunsFile keeps each lane's record in the trunk checkout's state, which a run from another checkout writes too.
const laneRunsFile = "lane_runs.json"

// ReadLaneRuns reads each lane's record kept at root; none where nothing is kept or it cannot be read.
func ReadLaneRuns(root string) map[string]LaneRecord { return readKept[LaneRecord](root, laneRunsFile) }

// RecordLaneRun keeps run as lane's latest, and as its latest pass where it passed, at root (lane-passes/do): a direct
// run of the lanes had kept no record of a lane's pass, its watch writing failures into a measurement run's record
// alone, so what a run proved was in its output only.
func RecordLaneRun(root, lane string, run LaneRun) error {
	runs := ReadLaneRuns(root)
	record := runs[lane]
	record.Last = run
	if run.Passed {
		record.Passed = run
	}
	runs[lane] = record
	return writeKept(root, laneRunsFile, runs)
}

// RecordLanePackages keeps run as the latest pass of each of packages, a lane's packages that passed in it, at root
// (lanes-package-pass): a lane that failed in one package had kept no pass of the others, so every close ran them all
// against the lane's last pass, days and dozens of commits old.
func RecordLanePackages(root, lane string, run LaneRun, packages []string) error {
	if len(packages) == 0 {
		return nil
	}
	runs := ReadLaneRuns(root)
	record := runs[lane]
	if record.Packages == nil {
		record.Packages = map[string]LaneRun{}
	}
	for _, pkg := range packages {
		record.Packages[pkg] = run
	}
	runs[lane] = record
	return writeKept(root, laneRunsFile, runs)
}

// readKept reads the record kept under file in root's state, by name; none where nothing is kept or it cannot be read.
func readKept[T any](root, file string) map[string]T {
	kept := map[string]T{}
	if jsonfile.Decode(filepath.Join(root, processlock.StateDirectory, file), &kept) != nil {
		return map[string]T{}
	}
	return kept
}

// writeKept keeps record under file in root's state.
func writeKept[T any](root, file string, record map[string]T) error {
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	return jsonfile.Write(filepath.Join(root, processlock.StateDirectory, file), record, runRecordMode)
}

// LanesDueEnvironment names, for a run of the lanes a runner starts, the file that keeps the commit due next (the
// runner's lanes_due.json): between packages the run yields to a newer commit named there (lanes-first/yield), and
// writes LanesYieldedFile beside it, which the runner reads to say the run yielded rather than failed.
const LanesDueEnvironment = "OVERGO_LANES_DUE"

// LanesYieldedFile is the record a run of the lanes that yielded writes beside the commit due: the commit it ran at
// and the newer commit it yielded to.
const LanesYieldedFile = "lanes_yielded.json"

// LanesYielded is a run of the lanes that yielded: the commit it ran at, the commit due it yielded to, and the lanes it
// failed before it yielded.
type LanesYielded struct {
	Commit string `json:"commit"`
	To     string `json:"to"`
	// Failures are the lanes that failed before the run yielded, which its verdict says beside the yield (SQA-437).
	Failures []string `json:"failures,omitempty"`
}
