package plan

import "time"

// TestPass is where a test last passed: the commit the run that passed it read, and when that run ended.
type TestPass struct {
	Commit string    `json:"commit"`
	Ended  time.Time `json:"ended"`
}

// testPassesFile keeps each test's last pass in the trunk checkout's state, beside the lanes' record, which a landing
// and a run of the cycle's lanes from another checkout write too.
const testPassesFile = "test_passes.json"

// PassedTest names a test as its pass is kept: its package's import path, a space, and its name.
func PassedTest(pkg, test string) string { return pkg + " " + test }

// ReadTestPasses reads each test's last pass kept at root, by PassedTest; none where nothing is kept or it cannot be
// read.
func ReadTestPasses(root string) map[string]TestPass { return readKept[TestPass](root, testPassesFile) }

// RecordTestPasses keeps pass as the last pass of each test named (PassedTest) at root (single-source/claim-record): a
// claim's test had been run again by every landing whose change moved a source the claim bound, in the fix stage and
// in an environment of its own, to rebind it; its last pass, wherever it ran, is now what a claim reads.
func RecordTestPasses(root string, pass TestPass, tests []string) error {
	if len(tests) == 0 {
		return nil
	}
	passes := ReadTestPasses(root)
	for _, test := range tests {
		passes[test] = pass
	}
	return writeKept(root, testPassesFile, passes)
}
