//go:build windows && modeltest

package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// localWorkerModel is the model the local worker is first held to: the judge's weights, served for text.
const localWorkerModel = "Qwen3.8-27B-UD-Q6_K"

// localFixture is a checkout with one step to work: Sum skips the first number, its test fails, and a fix of one line
// passes it. Its cmd/plan stands in for overgo's: -verify runs the step's check and -prompt shows the step, as the
// worker's run_check and show_step tools ask of it.
var localFixture = map[string]string{
	"go.mod": "module fixture\n\ngo 1.26\n",
	"calc/calc.go": `package calc

// Sum is the sum of every number it is given.
func Sum(numbers []int) int {
	total := 0
	for _, number := range numbers[1:] {
		total += number
	}
	return total
}
`,
	"calc/calc_test.go": `package calc

import "testing"

func TestSumAddsEveryNumber(t *testing.T) {
	if got := Sum([]int{1, 2, 3}); got != 6 {
		t.Fatalf("Sum(1, 2, 3) = %d, want 6", got)
	}
}
`,
	"cmd/plan/main.go": `package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
)

const step = "TASK calc/sum: Sum returns the sum of every number it is given, and TestSumAddsEveryNumber fails today\n" +
	"VERIFY go test ./calc -count=1\n" +
	"FILES calc: the change writes these and nothing else"

func main() {
	verify := flag.Bool("verify", false, "run the step's check")
	prompt := flag.Bool("prompt", false, "show the step")
	flag.Parse()
	switch {
	case *verify:
		check := exec.Command("go", "test", "./calc", "-count=1")
		check.Stdout, check.Stderr = os.Stdout, os.Stderr
		if err := check.Run(); err != nil {
			fmt.Println("the check failed:", err)
			os.Exit(1)
		}
		fmt.Println("the check passed")
	case *prompt:
		fmt.Println(step)
	}
}
`,
}

// TestALocalModelWorksAFixtureStep holds the local worker to working a step on overgo's own inference: Qwen3.8-27B,
// served by overgo's server, is sent the step and the six tools over the chat route, its tool calls are run in a fixture
// checkout, and the step's check, which fails before, passes after. Its replies and tool calls are stated.
func TestALocalModelWorksAFixtureStep(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	target, name := servedWorkerModel(t)

	checkout := t.TempDir()
	for file, text := range localFixture {
		path := filepath.Join(checkout, filepath.FromSlash(file))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(text), 0o644))
	}
	if _, err := runTool("git", "-C", checkout, "init", "-q"); err != nil {
		t.Fatalf("the fixture checkout is no repository: %v", err)
	}
	t.Chdir(checkout)
	if output, err := runTool("go", "run", "./cmd/plan", "-verify"); err == nil {
		t.Fatalf("the fixture step's check passes before any change:\n%s", output)
	}
	prompt := testcheck.Must(runTool("go", "run", "./cmd/plan", "-prompt"))(t)
	var log strings.Builder
	step := planStep(".")
	step.files = func() ([]string, error) { return []string{"calc"}, nil }
	work, err := workLocally(ctx, target, name, prompt, workerTools(".", step), &log)
	t.Logf("%s worked the step in %d replies and %d tool calls:\n%s", name, work.Turns, work.Calls, log.String())
	testcheck.NoError(t, err)
	if output, err := runTool("go", "run", "./cmd/plan", "-verify"); err != nil {
		t.Fatalf("the fixture step's check fails after the local worker: %v\n%s", err, output)
	}
}

// servedWorkerModel serves localWorkerModel through overgo's server, built from the checkout, for the test's life, and
// gives its address and served name.
func servedWorkerModel(t *testing.T) (target, name string) {
	t.Helper()
	ctx := t.Context()
	repository := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(repository))(t)
	db := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	entries, _, err := discovery.CapabilityCatalog(ctx, db, discovery.CatalogBound, discovery.LoadMemo(ctx, db))
	testcheck.NoError(t, errors.Join(err, db.Close()))
	var location string
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if capability.Task == recipe.TaskInference && capability.Stale == "" && entry.Present && strings.Contains(filepath.Base(entry.Location), localWorkerModel) {
				location = entry.Location
			}
		}
	}
	if location == "" {
		testskip.NotApplicable(t, "no servable "+localWorkerModel+" with current evidence is registered here")
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: []string{"build", "-o", binary, "overgo/cmd/server"}, Dir: repository, Stdout: os.Stderr, Stderr: os.Stderr})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("the server binary did not build: %v (exit %d)", err, receipt.ExitCode)
	}
	supervisor := testcheck.Must(modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: roots.Store, Dir: repository}, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, supervisor.Close()) })
	name = filepath.Base(location)
	lease := testcheck.Must(supervisor.Acquire(ctx, modelswap.Servable{Name: name, Location: location}))(t)
	t.Cleanup(lease.Release)
	return lease.Target, name
}
