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

// LaneRecord is a lane's latest run and its latest pass, which may be the same run.
type LaneRecord struct {
	Last   LaneRun `json:"last"`
	Passed LaneRun `json:"passed,omitzero"`
}

// laneRunsFile keeps each lane's record in the trunk checkout's state, which a run from another checkout writes too.
const laneRunsFile = "lane_runs.json"

// ReadLaneRuns reads each lane's record kept at root; none where nothing is kept or it cannot be read.
func ReadLaneRuns(root string) map[string]LaneRecord {
	runs := map[string]LaneRecord{}
	if jsonfile.Decode(filepath.Join(root, processlock.StateDirectory, laneRunsFile), &runs) != nil {
		return map[string]LaneRecord{}
	}
	return runs
}

// RecordLaneRun keeps run as lane's latest, and as its latest pass where it passed, at root (lane-passes/do): a direct
// run of the lanes had kept no record of a lane's pass, its watch writing failures into a measurement run's record
// alone, so what a run proved was in its output only.
func RecordLaneRun(root, lane string, run LaneRun) error {
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	runs := ReadLaneRuns(root)
	record := runs[lane]
	record.Last = run
	if run.Passed {
		record.Passed = run
	}
	runs[lane] = record
	return jsonfile.Write(filepath.Join(root, processlock.StateDirectory, laneRunsFile), runs, runRecordMode)
}
