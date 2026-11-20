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
// and none for the second package's other test; and the run keeps its yield with no failure (SQA-437).
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
	var kept []string
	var to string
	lanes := Plan{
		Suites: []SelectedSuite{{Lane: "models", Arguments: []string{"test", "-count=1", "./first", "./second"}, Environment: []string{"OVERGO_CUDA_TEST=1"},
			Each: [][]string{{"test", "-count=1", "./first"}, {"test", "-count=1", "./second"}}}},
		HoldsCard: func(_ string, work func() error) error { turns++; return work() },
		Yields: func() (string, bool) {
			asked++
			return "newer", asked > 1
		},
		Yielded: func(yielded string, failures []string) error { to, kept = yielded, failures; return nil },
	}
	var out bytes.Buffer
	if err := executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes); err == nil {
		t.Fatalf("a run that yielded ended as passed; want it yielded\n%s", out.String())
	}
	if turns != 2 || to != "newer" || len(kept) != 0 {
		t.Errorf("a run that yielded at its second package took %d turns of the card and kept its yield to %q with failures %q; want two turns and the yield with none\n%s",
			turns, to, kept, out.String())
	}
}
