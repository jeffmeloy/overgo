package processcontrol

import (
	"bytes"
	"context"
	"errors"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/processmeasure"
)

// TestSupervisorProcessTreeContract holds the supervisor to its
// contract: a completed command yields a receipt with its exit code
// and drained output; termination takes down the WHOLE tree including
// a grandchild the direct child spawned -- proven by the announced
// grandchild's death -- and Wait never reports success while
// anything still runs; the caller's cancellation terminates the tree and
// surfaces as an error with the receipt intact.
func TestSupervisorProcessTreeContract(t *testing.T) {
	ctx := t.Context()

	var out bytes.Buffer
	echo := shellCommand(t, "echo supervised")
	echo.Stdout = &out
	completed, err := Start(ctx, echo)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := completed.Wait(ctx)
	if err != nil || receipt.ExitCode != 0 || receipt.TreeTerminated {
		t.Fatalf("completed receipt = (%+v, %v)", receipt, err)
	}
	if receipt.StdoutBytes == 0 || !bytes.Contains(out.Bytes(), []byte("supervised")) {
		t.Fatalf("stdout not drained: %d bytes %q", receipt.StdoutBytes, out.String())
	}

	announced := WatchLine(nil, grandchildAnnouncement)
	spawn := grandchildCommand(t)
	spawn.Stdout = announced
	tree, err := Start(ctx, spawn)
	if err != nil {
		t.Fatal(err)
	}
	// The shell announces its child's pid once that child runs.
	grandchild, err := strconv.Atoi(strings.TrimSpace(<-announced.Line()))
	if err != nil || grandchild <= 0 || !ProcessAlive(grandchild) {
		t.Fatalf("grandchild announcement: pid %d, %v", grandchild, err)
	}
	if err := tree.Terminate(); err != nil {
		t.Fatal(err)
	}
	receipt, err = tree.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.TreeTerminated {
		t.Fatalf("termination not receipted: %+v", receipt)
	}
	if ProcessAlive(grandchild) {
		t.Fatalf("grandchild %d survived tree termination", grandchild)
	}

	// The caller's cancellation, not a deadline, ends a wait on a tree
	// that has not exited; the tree is terminated and the receipt kept.
	bounded, cancel := context.WithCancelCause(ctx)
	cancel(errors.New("the caller ended the wait"))
	long, err := Start(ctx, sleepCommand(t))
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = long.Wait(bounded)
	if err == nil {
		t.Fatal("cancellation reported success")
	}
	if receipt.WallNS == 0 {
		t.Fatalf("deadline receipt lacks wall evidence: %+v", receipt)
	}
}

// TestIsolatedModelProcess exposes the process-tree contract under the exact
// capability verifier identity used by the upgrade plan.
func TestIsolatedModelProcess(t *testing.T) {
	TestSupervisorProcessTreeContract(t)
}

func TestWrappedErrorClassification(t *testing.T) {
	command := shellCommand(t, "exit 7")
	supervised, err := Start(t.Context(), command)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := supervised.Wait(t.Context())
	if err != nil {
		t.Fatalf("nonzero process was treated as supervisor failure: %v", err)
	}
	if receipt.ExitCode != 7 || receipt.TreeTerminated {
		t.Fatalf("nonzero receipt = %+v", receipt)
	}
}

func TestSupervisorRetainsMeasurementFailure(t *testing.T) {
	supervised, err := Start(t.Context(), shellCommand(t, "echo retained"))
	if err != nil {
		t.Fatal(err)
	}
	// Invalidate only this process's clock; no global counter hook or sleep.
	supervised.clock = processmeasure.Stopwatch{}
	receipt, firstErr := supervised.Wait(t.Context())
	if firstErr == nil || receipt.ExitCode != 0 || receipt.StdoutBytes == 0 || !supervised.Exited() {
		t.Fatalf("terminal receipt/failure lost: %+v, %v", receipt, firstErr)
	}
	again, againErr := supervised.Wait(t.Context())
	if again != receipt || !errors.Is(againErr, firstErr) {
		t.Fatalf("repeated Wait changed outcome: %+v, %v; want %+v, %v", again, againErr, receipt, firstErr)
	}
}

func shellCommand(t *testing.T, script string) Command {
	t.Helper()
	if runtime.GOOS == "windows" {
		return Command{Path: "cmd", Args: []string{"/c", script}}
	}
	return Command{Path: "sh", Args: []string{"-c", script}}
}

func sleepCommand(t *testing.T) Command {
	t.Helper()
	if runtime.GOOS == "windows" {
		return Command{Path: "cmd", Args: []string{"/c", "ping -n 60 127.0.0.1 >nul"}}
	}
	return Command{Path: "sh", Args: []string{"-c", "sleep 60"}}
}

// grandchildAnnouncement prefixes the line the shell writes with its
// CHILD's pid, so tree termination is observable one level below the
// supervised command through that process's liveness.
const grandchildAnnouncement = "grandchild="

// grandchildCommand builds a shell that spawns a long-lived child,
// announces the child's pid, and then waits on it.
func grandchildCommand(t *testing.T) Command {
	t.Helper()
	if runtime.GOOS == "windows" {
		spawn := "$child = Start-Process -WindowStyle Hidden -PassThru cmd -ArgumentList '/c','ping -n 600 127.0.0.1 > nul'; Write-Output ('" + grandchildAnnouncement + "' + $child.Id); Wait-Process -Id $child.Id"
		return Command{Path: "powershell", Args: []string{"-NoProfile", "-Command", spawn}}
	}
	return Command{Path: "sh", Args: []string{"-c", "sleep 600 & echo " + grandchildAnnouncement + "$!; wait"}}
}
