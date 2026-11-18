package landing

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// laneTurnHolderEnvironment carries the lane turn's name to a process of this test binary that holds it
// (TestLaneTurnHolder).
const laneTurnHolderEnvironment = "OVERGO_TEST_LANE_TURN_HOLDER"

// TestLaneTurnHolder is a process of heldTurn: it holds the turn its environment names, says any wait for it and that
// it holds it, and gives it up when its parent closes its input. It passes at once in any other run.
func TestLaneTurnHolder(t *testing.T) {
	name := os.Getenv(laneTurnHolderEnvironment)
	if name == "" {
		return
	}
	release := testcheck.Must(processcontrol.TakeTurn(name, "a lane turn's holder", os.Stdout))(t)
	fmt.Println("taken")
	_, err := io.Copy(io.Discard, os.Stdin)
	testcheck.NoError(t, errors.Join(err, release()))
}

// heldTurn holds the lane turn for a test's goroutine, from a process of its own as a run holds it: it closes held once
// the turn is the holder's and keeps it until let closes. A turn held in the test's own process was ridden by the
// landing it ran, which then went on without waiting, so the commit a test made "during the wait" raced it.
func heldTurn(t *testing.T, held, let chan struct{}) func() {
	return func() {
		name, err := laneTurn()
		if err != nil {
			t.Error(err)
			close(held)
			return
		}
		holder := exec.Command(os.Args[0], "-test.run=^TestLaneTurnHolder$")
		holder.Env = append(os.Environ(), laneTurnHolderEnvironment+"="+name)
		input := testcheck.Must(holder.StdinPipe())(t)
		output := testcheck.Must(holder.StdoutPipe())(t)
		if err := holder.Start(); err != nil {
			t.Error(err)
			close(held)
			return
		}
		lines := bufio.NewScanner(output)
		for lines.Scan() && lines.Text() != "taken" {
		}
		close(held)
		<-let
		if err := errors.Join(input.Close(), holder.Wait()); err != nil {
			t.Error(err)
		}
	}
}

// heldFile is what the checkout holds at the slash path name, or nothing where it holds no such file.
func heldFile(checkout *sliceFixture, name string) string {
	data, err := os.ReadFile(filepath.Join(checkout.repo, filepath.FromSlash(name)))
	if err != nil {
		return ""
	}
	return string(data)
}

// packageSnapshot is a syntax snapshot of one package file, internal/p/p.go, holding source.
func packageSnapshot(t *testing.T, source string) gosyntax.Snapshot {
	t.Helper()
	result := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)}))(t)
	return result
}
