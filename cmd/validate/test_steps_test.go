package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testskip"
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

// TestALaneRunTestByTestIsJudgedAsItsPackages holds a lane run test by test to the verdict its packages earned run whole
// (lanes-verdict): a package whose test passed beside one declaring itself inapplicable passes, a package whose test
// failed beside such a test fails and is not reported inapplicable, a package of declared exclusions alone is
// inapplicable, and a pass in one package lends another nothing. Judged test by test, the first had been an
// inapplicable lane, which withheld every cell's measurement, and the second an inapplicable lane hiding its failure
// (SQA-417).
func TestALaneRunTestByTestIsJudgedAsItsPackages(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644))
	elsewhere := fmt.Sprintf("func TestElsewhere(t *testing.T) {\n\tt.Attr(%q, %q)\n\tt.Skip(%q)\n}\n", testskip.Key, testskip.KindInapplicable, testskip.Inapplicable+": a fixture of another lane")
	for pkg, tests := range map[string]string{
		"passes":   "func TestPasses(t *testing.T) {}\n\n" + elsewhere,
		"fails":    "func TestFails(t *testing.T) { t.Fatal(\"broken\") }\n\n" + elsewhere,
		"declared": elsewhere,
	} {
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, pkg), 0o755))
		source := "package " + pkg + "\n\nimport \"testing\"\n\n" + tests
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, pkg, pkg+"_test.go"), []byte(source), 0o644))
	}
	lane := func(name string, packages ...string) SelectedSuite {
		suite := SelectedSuite{Lane: name, Arguments: slices.Concat([]string{"test", "-count=1"}, packages)}
		for _, pkg := range packages {
			suite.Each = append(suite.Each, []string{"test", "-count=1", pkg})
		}
		return suite
	}
	lanes := Plan{Suites: []SelectedSuite{
		lane("passing", "./passes"), lane("failing", "./fails"), lane("declared", "./declared"), lane("lent", "./passes", "./declared"),
	}}
	var out bytes.Buffer
	err := executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes)
	if err == nil {
		t.Fatalf("the lanes passed; want the failing lane failed and two inapplicable\n%s", out.String())
	}
	failed, inapplicable, _ := strings.Cut(err.Error(), errInapplicable.Error())
	for lane, want := range map[string]bool{"passing": false, "failing": true, "declared": false, "lent": false} {
		if strings.Contains(failed, "cycle lane "+lane+":") != want {
			t.Errorf("cycle lane %s is reported failed %v, want %v: %v", lane, !want, want, err)
		}
	}
	for lane, want := range map[string]bool{"passing": false, "failing": false, "declared": true, "lent": true} {
		if strings.Contains(inapplicable, "cycle lane "+lane+":") != want {
			t.Errorf("cycle lane %s is reported inapplicable %v, want %v: %v", lane, !want, want, err)
		}
	}
	if !strings.Contains(failed, "re-acquired 1 cell(s)") || !strings.Contains(failed, "TestFails") {
		t.Errorf("the lanes run = %v; want the passing lane acquired and the failing lane's failure named", err)
	}
}
