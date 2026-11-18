//go:build windows

package processcontrol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// detachedProbeEnvironment carries a detached probe to a process of this test binary (TestDetachedProcess).
const detachedProbeEnvironment = "OVERGO_TEST_DETACHED"

// detachedProbe is one process's part: start the waiter detached and end, or wait for the turn the test holds and then
// write the file that says it outlived its starter's tree.
type detachedProbe struct{ Part, Turn, Outlived string }

// TestDetachedProcess is a process of TestADetachedProcessOutlivesTheTreeThatStartedIt, and passes at once in any other
// run.
func TestDetachedProcess(t *testing.T) {
	encoded := os.Getenv(detachedProbeEnvironment)
	if encoded == "" {
		return
	}
	var probe detachedProbe
	testcheck.NoError(t, json.Unmarshal([]byte(encoded), &probe))
	switch probe.Part {
	case "start":
		waiter := probe
		waiter.Part = "wait"
		pid := testcheck.Must(StartDetached(detachedCommand(t, waiter)))(t)
		fmt.Println(pid)
	case "wait":
		testcheck.NoError(t, ResourceTransaction(probe.Turn, func() error { return os.WriteFile(probe.Outlived, []byte("outlived"), 0o600) }))
	}
}

// detachedCommand is a process of this test binary that plays probe's part.
func detachedCommand(t *testing.T, probe detachedProbe) Command {
	t.Helper()
	encoded := testcheck.Must(json.Marshal(probe))(t)
	return Command{
		Path: os.Args[0], Args: []string{"-test.run=^TestDetachedProcess$", "-test.timeout=0"},
		Env: append(os.Environ(), detachedProbeEnvironment+"="+string(encoded)),
	}
}

// TestADetachedProcessOutlivesTheTreeThatStartedIt holds StartDetached to a process that outlives the supervised tree
// that started it (lanes-on-close): a process started detached by one Run supervises still runs once Run's tree has
// ended and its job has closed, which ends every process left in it, and does its work after. A closing landing's lanes
// runner outlives the landing that starts it.
func TestADetachedProcessOutlivesTheTreeThatStartedIt(t *testing.T) {
	t.Parallel()
	probe := detachedProbe{Part: "start", Turn: fmt.Sprintf("overgo-test-detached-%d", os.Getpid()), Outlived: filepath.Join(t.TempDir(), "outlived")}
	entered, leave, left := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		left <- ResourceTransaction(probe.Turn, func() error { close(entered); <-leave; return nil })
	}()
	<-entered
	var said bytes.Buffer
	command := detachedCommand(t, probe)
	command.Stdout, command.Stderr = &said, &said
	receipt, err := Run(t.Context(), command)
	if err != nil || receipt.ExitCode != 0 {
		close(leave)
		t.Fatalf("the starter ended %d, %v:\n%s", receipt.ExitCode, err, said.String())
	}
	line, _, _ := strings.Cut(said.String(), "\n")
	pid := testcheck.Must(strconv.Atoi(strings.TrimSpace(line)))(t)
	waiter, err := os.FindProcess(pid)
	close(leave)
	testcheck.NoError(t, errors.Join(err, <-left))
	state := testcheck.Must(waiter.Wait())(t)
	if written, err := os.ReadFile(probe.Outlived); err != nil || state.ExitCode() != 0 || string(written) != "outlived" {
		t.Fatalf("the detached process ended %d with %q, %v, once its starter's tree had ended; want it to do its work", state.ExitCode(), written, err)
	}
}
