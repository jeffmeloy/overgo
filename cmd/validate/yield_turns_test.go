package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAYieldTakesNoMoreTurns holds a run of the lanes that yields to taking no turn for the steps after its yield
// (SQA-438): a lane on the card of two fixture packages, the second of two tests, whose newer commit is named due as
// it turns to the second, takes the card's turn for the first package's test and for the step that finds the yield,
// and none for the second package's other test; and the run ends with no failure (SQA-437, lanes-record).
func TestAYieldTakesNoMoreTurns(t *testing.T) {
	t.Parallel()
	root, _ := fixtureRepository(t)
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644))
	tests := map[string]string{"first": "func TestOne(t *testing.T) {}\n", "second": "func TestTwo(t *testing.T) {}\n\nfunc TestThree(t *testing.T) {}\n"}
	for pkg, body := range tests {
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, pkg), 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, pkg, pkg+"_test.go"), []byte("package "+pkg+"\n\nimport \"testing\"\n\n"+body), 0o644))
	}
	turns, asked := 0, 0
	lanes := Plan{
		Suites: []SelectedSuite{{Lane: "models", Arguments: []string{"test", "-count=1", "./first", "./second"}, Environment: []string{"OVERGO_CUDA_TEST=1"},
			Each: [][]string{{"test", "-count=1", "./first"}, {"test", "-count=1", "./second"}}}},
		HoldsCard: func(_ string, work func() error) error { turns++; return work() },
		Yields: func() (string, bool) {
			asked++
			return "newer", asked > 1
		},
	}
	var out bytes.Buffer
	err := executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes)
	if turns != 2 || err != nil {
		t.Errorf("a run that yielded at its second package took %d turns of the card and ended %v; want two turns and no failure\n%s", turns, err, out.String())
	}
}
