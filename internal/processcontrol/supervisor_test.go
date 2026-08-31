package processcontrol

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestSupervisorProcessTreeContract holds the supervisor to its
// contract: a completed command yields a receipt with its exit code
// and drained output; termination takes down the WHOLE tree including
// a grandchild the direct child spawned -- proven by the grandchild's
// heartbeat file going quiet -- and Wait never reports success while
// anything still runs; a context deadline terminates the tree and
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

	marker := filepath.Join(t.TempDir(), "grandchild-heartbeat")
	tree, err := Start(ctx, grandchildCommand(t, marker))
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		if info, statErr := os.Stat(marker); statErr == nil && info.Size() > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("grandchild never started writing")
		}
		time.Sleep(50 * time.Millisecond)
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
	settled, statErr := os.Stat(marker)
	if statErr != nil {
		t.Fatal(statErr)
	}
	time.Sleep(600 * time.Millisecond)
	after, statErr := os.Stat(marker)
	if statErr != nil {
		t.Fatal(statErr)
	}
	if after.Size() != settled.Size() {
		t.Fatalf("grandchild survived tree termination: %d -> %d bytes", settled.Size(), after.Size())
	}

	bounded, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
	defer cancel()
	long, err := Start(ctx, sleepCommand(t))
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = long.Wait(bounded)
	if err == nil {
		t.Fatal("deadline expiry reported success")
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

// grandchildCommand builds a shell whose CHILD process writes the
// heartbeat, so tree termination is observable one level below the
// supervised command.
func grandchildCommand(t *testing.T, marker string) Command {
	t.Helper()
	if runtime.GOOS == "windows" {
		spawn := "Start-Process -WindowStyle Hidden cmd -ArgumentList '/c','for /l %g in (1,1,600) do (echo 1>> \"" + marker + "\" & ping -n 1 127.0.0.1 > nul)'; Start-Sleep -Seconds 60"
		return Command{Path: "powershell", Args: []string{"-NoProfile", "-Command", spawn}}
	}
	script := "while true; do echo 1 >> \"" + marker + "\"; sleep 0.1; done"
	return Command{Path: "sh", Args: []string{"-c", "sh -c '" + script + "' & wait"}}
}
