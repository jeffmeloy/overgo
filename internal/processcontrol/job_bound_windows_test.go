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

// TestJobChildProcess is a child of the job bound tests (jobChildren): one that commits a little and ends, or one that
// commits without end. It passes at once in any other run.
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

// jobChildren are the children of one test, a passing one and a runaway (TestJobChildProcess), whose commands are that
// test's own among the package's tests by a nonce, and whose output the test reads in said.
func jobChildren(said *strings.Builder) func(part string) Command {
	nonce := fmt.Sprintf("^Nonce%d_%d$", os.Getpid(), time.Now().UnixNano())
	// A child that outlives its minute ends itself, so a bound that holds it without ending it fails the test.
	return func(part string) Command {
		return Command{Path: os.Args[0], Args: []string{"-test.run=^TestJobChildProcess$|" + nonce + part, "-test.timeout=1m"},
			Env: append(os.Environ(), jobChildEnvironment+"="+part), Stderr: said}
	}
}

// jobKeyOf is the key a command's need is kept by (jobKey).
func jobKeyOf(command Command) string {
	return jobKey(exec.Command(command.Path, command.Args...))
}

// TestARunawayChildIsEndedAtTheMachinesFreeMemory holds a supervised tree to its bound (check-memory/do): a replayed
// check run on the fault its step cured took 279 GB of commit at 10:05 on 2026-11-15, and a served model's read failed
// beside it. A command this process has not run to a pass is bound by the commit free as it starts, and its passing
// run records its need, its whole tree's peak; its next run is bound by that need and the commit free as it starts.
func TestARunawayChildIsEndedAtTheMachinesFreeMemory(t *testing.T) {
	t.Parallel()
	var said strings.Builder
	passing := jobChildren(&said)("pass")
	first := testcheck.Must(Run(t.Context(), passing))(t)
	need, recorded := readJobNeed(jobKeyOf(passing))
	if first.ExitCode != 0 || first.MemoryLimit == 0 || !recorded || need < 32<<20 {
		t.Fatalf("a command's first run read %+v and recorded a need of %d (%t); want it bound by the commit free, passing, its need at least the 32 MiB it held", first, need, recorded)
	}
	second := testcheck.Must(Run(t.Context(), passing))(t)
	if second.ExitCode != 0 || second.MemoryLimit <= need {
		t.Fatalf("a command's run after its need was recorded read %+v; want it bound past its need of %d by the commit free", second, need)
	}
}

// TestAFirstSeenRunawayIsEndedAtTheMachinesFreeMemory holds a command's first run to a bound (first-bound/do, SQA-404):
// the runaway child, a command no run of which passed, is bound as boundOf reads it, by the commit free as it starts,
// and ended at that bound, its run saying no pass of its command recorded a need, and records none. A first run left
// unbounded is refused before it starts, so the check never lets a runaway take the machine; and the bound boundOf
// gives is held to 256 MiB here, so the child ends where this machine's free commit is not spent.
func TestAFirstSeenRunawayIsEndedAtTheMachinesFreeMemory(t *testing.T) {
	t.Parallel()
	var said strings.Builder
	runaway := jobChildren(&said)("runaway")
	const held = 256 << 20
	runaway.bounded = func(command *exec.Cmd) (jobBound, error) {
		bound, err := boundOf(command)
		if err == nil && bound.limit == 0 {
			err = fmt.Errorf("a command no run of which passed is bound by nothing: its limit is %d bytes", bound.limit)
		}
		bound.free, bound.limit = min(bound.free, held), min(bound.limit, held)
		return bound, err
	}
	supervised := testcheck.Must(Start(t.Context(), runaway))(t)
	receipt, err := supervised.Wait(t.Context())
	if receipt.ExitCode == 0 || !receipt.MemoryLimited || receipt.MemoryLimit != held || err == nil || !strings.Contains(err.Error(), "no run of its command passed") {
		t.Fatalf("a first-seen child committing without end ended %+v, %v; want it ended at its bound, its run saying no pass recorded its need; it said:\n%.3000s", receipt, err, said.String())
	}
	if _, recorded := readJobNeed(jobKeyOf(runaway)); recorded {
		t.Error("a run that reached its bound recorded a need")
	}
}
