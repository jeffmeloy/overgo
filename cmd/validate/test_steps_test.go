package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALanePackageRunsTestByTest holds a lane run package by package to running each package's tests one at a time
// (lanes-yield): a fixture package of two tests is two steps of the lane, each its own go test process naming one test
// with the lane's flags, so the turn the steps take (Plan.TakesTurns, landing.HoldTestTurnEach) passes to a waiting
// commit between the tests. In the run at c79cfc121 one package held the card and the test turn 1 h 26 min, and a
// commit waited 81 min.
func TestALanePackageRunsTestByTest(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644))
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "p"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "p", "p_test.go"), []byte(`package p

import (
	"fmt"
	"os"
	"testing"
)

func record(t *testing.T) {
	file, err := os.OpenFile(os.Getenv("LANE_FIXTURE_RECORD"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	fmt.Fprintln(file, t.Name(), os.Getpid())
}

func TestOne(t *testing.T) { record(t) }

func TestTwo(t *testing.T) { record(t) }
`), 0o644))
	record := filepath.Join(t.TempDir(), "record.txt")
	steps := 0
	lanes := Plan{
		Suites: []SelectedSuite{{
			Lane: "models", Arguments: []string{"test", "-count=1", "./p"}, Each: [][]string{{"test", "-count=1", "./p"}},
			Environment: []string{"LANE_FIXTURE_RECORD=" + record},
		}},
		TakesTurns: func(_ string, outer func(func() error) error, each []func() error) error {
			steps = len(each)
			for _, step := range each {
				testcheck.NoError(t, outer(step))
			}
			return nil
		},
	}
	var out bytes.Buffer
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes))
	ran := strings.Fields(string(testcheck.Must(os.ReadFile(record))(t)))
	if steps != 2 || len(ran) != 4 || ran[0] != "TestOne" || ran[2] != "TestTwo" || ran[1] == ran[3] {
		t.Fatalf("the lane took %d steps and its tests ran as %q; want TestOne and TestTwo, each a step and a process of its own\n%s", steps, ran, out.String())
	}
}
