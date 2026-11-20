//go:build windows

package processcontrol

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// jobChildEnvironment names the part a process of this test binary plays (TestJobChildProcess).
const jobChildEnvironment = "OVERGO_TEST_JOB_CHILD"

// TestJobChildProcess is a child of the job bound tests (runawayChild), one that commits without end. It passes at once
// in any other run.
func TestJobChildProcess(t *testing.T) {
	if os.Getenv(jobChildEnvironment) != "runaway" {
		return
	}
	var held [][]byte
	for {
		held = append(held, touched(16<<20))
	}
}

// touched is size bytes written through, so each page is committed.
func touched(size int) []byte {
	block := make([]byte, size)
	for at := 0; at < size; at += 4096 {
		block[at] = 1
	}
	return block
}

// runawayChild is a child that commits without end (TestJobChildProcess), whose output goes to said. A child that
// outlives its minute ends itself, so a bound that holds it without ending it fails the test.
func runawayChild(said *strings.Builder) Command {
	return Command{Path: os.Args[0], Args: []string{"-test.run=^TestJobChildProcess$", "-test.timeout=1m"},
		Env: append(os.Environ(), jobChildEnvironment+"=runaway"), Stderr: said}
}

// TestARunawayChildIsEndedAtItsBound holds a supervised tree to its bound (check-memory/do): a replayed check run on the
// fault its step cured took 279 GB of commit at 10:05 on 2026-11-15, and a served model's read failed beside it. The
// bound boundOf gives is held to 256 MiB here, so the child ends where this machine's commit is not spent, and its run
// says the bound it reached.
func TestARunawayChildIsEndedAtItsBound(t *testing.T) {
	t.Parallel()
	var said strings.Builder
	runaway := runawayChild(&said)
	const held = 256 << 20
	runaway.bounded = func(command *exec.Cmd) (jobBound, error) {
		bound, err := boundOf(command)
		bound.limit = min(bound.limit, held)
		return bound, err
	}
	receipt, err := testcheck.Must(Start(t.Context(), runaway))(t).Wait(t.Context())
	if receipt.ExitCode == 0 || !receipt.MemoryLimited || receipt.MemoryLimit != held || err == nil || !strings.Contains(err.Error(), "the machine's commit limit as it started") {
		t.Fatalf("a child committing without end ended %+v, %v; want it ended at its bound, its run saying the bound it reached; it said:\n%.3000s", receipt, err, said.String())
	}
}

// TestABoundIsVerdictIndependentOfNeighbours holds a tree's bound to the machine's commit limit or twice its memory,
// whatever the processes beside it hold (verdict-steady): bounded by the commit free as it started, the cycle's lanes at
// 1ab50872c ended at 20.7 GiB on 2026-11-19 while a model server another lane had started held the rest. A neighbour
// committing a GiB moves the commit free and leaves the bound where it was.
func TestABoundIsVerdictIndependentOfNeighbours(t *testing.T) {
	_, limit, err := MachineCommit()
	testcheck.NoError(t, err)
	before := testcheck.Must(boundOf(nil))(t)
	neighbour := touched(1 << 30)
	freeBeside, limitBeside, err := MachineCommit()
	testcheck.NoError(t, err)
	beside := testcheck.Must(boundOf(nil))(t)
	if before.limit != beside.limit || before.limit < limit || limitBeside != limit {
		t.Fatalf("a bound read %d bytes, then %d with a GiB committed beside it (%d free, the limit %d then %d); want it the same, at least the machine's commit limit",
			before.limit, beside.limit, freeBeside, limit, limitBeside)
	}
	_ = neighbour[0]
}

// TestABoundAdmitsAPageFileGrownAsWindowsGrowsIt holds a tree's bound to admitting what Flash-Next's open commits by
// design and to ending a runaway well before it starves the machine (flash-next-bound): every internal/inference test
// that opens Flash-Next's round reserves its commit with an ordinary commit, which Windows grows the page file for
// (flash-next-load), and its tree peaked at 115.6 GiB, ended in every lanes run since 2026-11-18 at the commit limit as
// it started, 99.6 GiB on a machine of 96 GiB at rest; a runaway check took 279 GB on 2026-11-15 (check-memory).
func TestABoundAdmitsAPageFileGrownAsWindowsGrowsIt(t *testing.T) {
	const flashNextTree, runaway = 1156 << 30 / 10, 279_000_000_000
	bound := testcheck.Must(boundOf(nil))(t)
	_, limit, err := MachineCommit()
	testcheck.NoError(t, err)
	if bound.limit < flashNextTree || bound.limit >= runaway || bound.limit < limit {
		t.Fatalf("a tree is bound at %.1f GiB beside a commit limit of %.1f GiB; want room for Flash-Next's %.1f GiB tree and under the runaway's %.1f GiB",
			float64(bound.limit)/(1<<30), float64(limit)/(1<<30), float64(flashNextTree)/(1<<30), float64(runaway)/(1<<30))
	}
}

// TestAStartCanceledAsItIsBoundMakesNoChild holds Start to its context up to the child's creation (SQA-424): a swap
// canceled while its launch was being sized had a server started for it, then stopped. A context canceled as the bound
// is read returns its cancellation, and no child.
func TestAStartCanceledAsItIsBoundMakesNoChild(t *testing.T) {
	var said strings.Builder
	runaway := runawayChild(&said)
	ctx, cancel := context.WithCancel(t.Context())
	runaway.bounded = func(command *exec.Cmd) (jobBound, error) {
		cancel()
		return jobBound{limit: 256 << 20}, nil
	}
	supervised, err := Start(ctx, runaway)
	if supervised != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("a start canceled as it was bound returned %v, %v; want its cancellation and no child", supervised, err)
	}
}
