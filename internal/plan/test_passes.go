package plan

import (
	"path/filepath"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// TestResult is a test's last result: the commit the run that ran it read, whether it failed, and when it ended. It is
// the one record of what the slow tests proved (lanes-record): a lane's pass and its packages' passes had been kept
// beside it, at units coarser than the test the run works in, so a run cut short kept no package pass and every close
// ran its tests again (TestMOSSGenerationCheckpoint in 7 runs on 2026-11-24 and 25).
type TestResult struct {
	Commit string    `json:"commit"`
	Failed bool      `json:"failed,omitzero"`
	Ended  time.Time `json:"ended"`
}

// testResultsFile keeps each test's last result in the trunk checkout's state, which a landing and a run of the cycle's
// lanes from another checkout write too; it kept passes alone before, which read as results that did not fail.
const testResultsFile = "test_passes.json"

// PassedTest names a test as its result is kept: its package's import path, a space, and its name; a lane run whole
// is kept as LaneTest names it.
func PassedTest(pkg, test string) string { return pkg + " " + test }

// LaneTest names a cycle lane run whole as its result is kept, beside the tests of the lanes run test by test.
func LaneTest(lane string) string { return "cycle lane " + lane }

// ReadTestResults reads each test's last result kept at root, by PassedTest; none where nothing is kept or it cannot
// be read.
func ReadTestResults(root string) map[string]TestResult {
	kept := map[string]TestResult{}
	if jsonfile.Decode(filepath.Join(root, processlock.StateDirectory, testResultsFile), &kept) != nil {
		return map[string]TestResult{}
	}
	return kept
}

// RecordTestResults keeps result as the last result of each test named (PassedTest) at root, as each ends, so a run
// cut short or yielded keeps what it ran (single-source/claim-record, lanes-record).
func RecordTestResults(root string, result TestResult, tests ...string) error {
	if len(tests) == 0 {
		return nil
	}
	results := ReadTestResults(root)
	for _, test := range tests {
		results[test] = result
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	return jsonfile.Write(filepath.Join(root, processlock.StateDirectory, testResultsFile), results, runRecordMode)
}
