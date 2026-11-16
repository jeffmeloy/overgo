//go:build windows

package processcontrol

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// jobChildEnvironment names the part a process of this test binary plays (TestJobChildProcess).
const jobChildEnvironment = "OVERGO_TEST_JOB_CHILD"

// TestJobChildProcess is a child of TestARunawayChildIsEndedAtTheMachinesFreeMemory: one that commits a little and
// ends, or one that commits without end. It passes at once in any other run.
func TestJobChildProcess(t *testing.T) {
	var held [][]byte
	switch os.Getenv(jobChildEnvironment) {
	case "pass":
		held = append(held, touched(32<<20))
	case "runaway":
		for {
			held = append(held, touched(16<<20))
		}
	}
	_ = held
}

// touched is size bytes written through, so each page is committed.
func touched(size int) []byte {
	block := make([]byte, size)
	for at := 0; at < size; at += 4096 {
		block[at] = 1
	}
	return block
}

// TestARunawayChildIsEndedAtTheMachinesFreeMemory holds a supervised tree to its bound (check-memory/do): a replayed
// check run on the fault its step cured took 279 GB of commit at 10:05 on 2026-11-15, and a served model's read failed
// beside it. A command this process has not run to a pass runs unbounded and its passing run records its need, its
// whole tree's peak; its next run is bound by that need and the commit free as it starts. A child that commits without
// end, bound by 256 MiB, is ended there, its run saying it reached its memory limit, and records no need.
func TestARunawayChildIsEndedAtTheMachinesFreeMemory(t *testing.T) {
	t.Parallel()
	// The nonce keeps the commands this test's own among the package's tests.
	nonce := fmt.Sprintf("^Nonce%d_%d$", os.Getpid(), time.Now().UnixNano())
	// A child that outlives its minute ends itself, so a bound that holds it without ending it fails the test.
	var said strings.Builder
	child := func(part string) Command {
		return Command{Path: os.Args[0], Args: []string{"-test.run=^TestJobChildProcess$|" + nonce + part, "-test.timeout=1m"},
			Env: append(os.Environ(), jobChildEnvironment+"="+part), Stderr: &said}
	}
	keyOf := func(command Command) string {
		return jobKey(exec.Command(command.Path, command.Args...))
	}
	passing, runaway := child("pass"), child("runaway")

	first := testcheck.Must(Run(t.Context(), passing))(t)
	need, recorded := readJobNeed(keyOf(passing))
	if first.ExitCode != 0 || first.MemoryLimit != 0 || !recorded || need < 32<<20 {
		t.Fatalf("a command's first run read %+v and recorded a need of %d (%t); want it unbounded, passing, its need at least the 32 MiB it held", first, need, recorded)
	}
	second := testcheck.Must(Run(t.Context(), passing))(t)
	if second.ExitCode != 0 || second.MemoryLimit <= need {
		t.Fatalf("a command's run after its need was recorded read %+v; want it bound past its need of %d by the commit free", second, need)
	}

	const bound = 256 << 20
	runaway.bounded = func(command *exec.Cmd) (jobBound, error) {
		return jobBound{key: jobKey(command), free: bound, limit: bound}, nil
	}
	supervised := testcheck.Must(Start(t.Context(), runaway))(t)
	receipt, err := supervised.Wait(t.Context())
	if receipt.ExitCode == 0 || !receipt.MemoryLimited || receipt.MemoryLimit != bound || err == nil || !strings.Contains(err.Error(), "reached its memory limit") {
		t.Fatalf("a child committing without end under a 256 MiB bound ended %+v, %v; want it ended at the bound, its run saying so; it said:\n%.3000s", receipt, err, said.String())
	}
	if _, recorded := readJobNeed(keyOf(runaway)); recorded {
		t.Error("a run that reached its bound recorded a need")
	}
}
