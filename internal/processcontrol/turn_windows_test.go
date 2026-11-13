//go:build windows

package processcontrol

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// turnProbeEnvironment carries a turn probe to a process of this test binary (TestTurnProcess).
const turnProbeEnvironment = "OVERGO_TEST_TURN"

// turnProbe is one process's part: hold the turn and answer its parent's lines, ride its holder's place, or take it.
type turnProbe struct{ Name, Purpose, Part string }

// TestTurnProcess is a process of TestOneCardTurnQueuesEveryUser, and passes at once in any other run.
func TestTurnProcess(t *testing.T) {
	encoded := os.Getenv(turnProbeEnvironment)
	if encoded == "" {
		return
	}
	var probe turnProbe
	testcheck.NoError(t, json.Unmarshal([]byte(encoded), &probe))
	switch probe.Part {
	case "hold":
		release := testcheck.Must(TakeTurn(probe.Name, probe.Purpose, os.Stdout))(t)
		fmt.Println("taken")
		input := bufio.NewScanner(os.Stdin)
		for input.Scan() {
			switch input.Text() {
			case "child":
				child := turnCommand(t, turnProbe{Name: probe.Name, Purpose: "the child", Part: "ride"})
				child.Stderr = os.Stderr
				said, _, _ := strings.Cut(string(testcheck.Must(child.Output())(t)), "\n")
				fmt.Println(said)
			case "give":
				testcheck.NoError(t, release())
				fmt.Println("given")
			case "count":
				testcheck.NoError(t, TurnCount(3, 5))
				fmt.Println("counted")
			}
		}
	case "ride":
		// A child that would wait for its own parent says so instead of waiting for ever.
		record, held, err := turnHolder(probe.Name)
		testcheck.NoError(t, err)
		if !held || heldTurns()[probe.Name] != record.Process {
			fmt.Println("would wait for its parent")
			return
		}
		var said strings.Builder
		release := testcheck.Must(TakeTurn(probe.Name, probe.Purpose, &said))(t)
		testcheck.NoError(t, release())
		fmt.Printf("rode%s\n", said.String())
	case "take":
		release := testcheck.Must(TakeTurn(probe.Name, probe.Purpose, os.Stdout))(t)
		fmt.Println("taken")
		testcheck.NoError(t, release())
	}
}

// turnCommand is a process of this test binary playing probe's part.
func turnCommand(t *testing.T, probe turnProbe) *exec.Cmd {
	t.Helper()
	encoded := testcheck.Must(json.Marshal(probe))(t)
	// The test's end ends the process.
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestTurnProcess$", "-test.timeout=0")
	command.Env = append(os.Environ(), turnProbeEnvironment+"="+string(encoded))
	return command
}

// turnProcess is a running turn probe: its input, and its output read a line at a time.
type turnProcess struct {
	command *exec.Cmd
	input   io.WriteCloser
	output  *bufio.Reader
}

func startTurnProcess(t *testing.T, probe turnProbe) turnProcess {
	t.Helper()
	command := turnCommand(t, probe)
	command.Stderr = os.Stderr
	process := turnProcess{command: command, input: testcheck.Must(command.StdinPipe())(t), output: bufio.NewReader(testcheck.Must(command.StdoutPipe())(t))}
	testcheck.NoError(t, command.Start())
	t.Cleanup(func() { _ = command.Wait() })
	return process
}

func (p turnProcess) line(t *testing.T) string {
	t.Helper()
	return strings.TrimSpace(testcheck.Must(p.output.ReadString('\n'))(t))
}

func (p turnProcess) say(t *testing.T, line string) {
	t.Helper()
	testcheck.Must(io.WriteString(p.input, line+"\n"))(t)
}

// testTurn names a turn of the test alone, its files removed when the test ends.
func testTurn(t *testing.T) string {
	t.Helper()
	name := "overgo-test-turn:" + t.Name() + ":" + fmt.Sprint(os.Getpid())
	t.Cleanup(func() {
		for _, kind := range []string{turnRecordFile, turnWaitingFile} {
			if path, err := turnFile(name, kind); err == nil {
				_ = os.Remove(path)
			}
		}
	})
	return name
}

// TestAWaiterSaysTheHoldersCount holds a waiter's line to its holder's own count (card-handover/counts): a holder that
// has counted 3 of 5 (TurnCount) is named by a waiter at 3/5 and the time it got there, where on 2026-11-12 a test's
// one line said only which render held the card and since when, and nothing of the 31 videos it drew meanwhile.
func TestAWaiterSaysTheHoldersCount(t *testing.T) {
	name := testTurn(t)
	holder := startTurnProcess(t, turnProbe{Name: name, Purpose: "a render", Part: "hold"})
	if got := holder.line(t); got != "taken" {
		t.Fatalf("the holder: %q, want taken", got)
	}
	holder.say(t, "count")
	if got := holder.line(t); got != "counted" {
		t.Fatalf("the holder: %q, want counted", got)
	}
	waiter := startTurnProcess(t, turnProbe{Name: name, Purpose: "a test", Part: "take"})
	if waiting := waiter.line(t); !strings.Contains(waiting, "for a render since") || !strings.Contains(waiting, ", at 3/5 reached ") {
		t.Fatalf("the waiter said %q, want its holder's count, 3/5, and when it got there", waiting)
	}
	holder.say(t, "give")
	if got := holder.line(t); got != "given" {
		t.Fatalf("the holder: %q, want given", got)
	}
	testcheck.NoError(t, waiter.command.Wait())
}

// TestOneCardTurnQueuesEveryUser holds the card turn's queue across three processes: the first takes the turn; the
// second waits for it, saying once which process holds it and for what; the first's child takes its parent's place
// without waiting, while the second still waits; and the second takes the turn as the first gives it up.
func TestOneCardTurnQueuesEveryUser(t *testing.T) {
	name := testTurn(t)
	first := startTurnProcess(t, turnProbe{Name: name, Purpose: "the first holder", Part: "hold"})
	if got := first.line(t); got != "taken" {
		t.Fatalf("the first process: %q, want taken", got)
	}
	second := startTurnProcess(t, turnProbe{Name: name, Purpose: "the second", Part: "take"})
	waiting := second.line(t)
	t.Logf("the waiter said: %s", waiting)
	for _, want := range []string{"waits for the turn " + name, fmt.Sprintf("(pid %d)", first.command.Process.Pid), "for the first holder since"} {
		if !strings.Contains(waiting, want) {
			t.Fatalf("the waiter said %q, which does not name %q", waiting, want)
		}
	}
	first.say(t, "child")
	if got := first.line(t); got != "rode" {
		t.Fatalf("the first process's child: %q, want it to ride its parent's place and say nothing", got)
	}
	first.say(t, "give")
	if got := first.line(t); got != "given" {
		t.Fatalf("the first process: %q, want given", got)
	}
	if took := second.line(t); !strings.Contains(took, "took the turn "+name+" after") {
		t.Fatalf("the waiter said %q after its wait, want how long it waited", took)
	}
	if got := second.line(t); got != "taken" {
		t.Fatalf("the waiter: %q, want taken, having said its wait once", got)
	}
	testcheck.NoError(t, second.command.Wait())
}
