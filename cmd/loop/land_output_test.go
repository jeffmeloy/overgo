package main

import (
	"io"
	"os"
	"strings"
	"testing"
)

const messageEcho = "OVERGO_TEST_ECHO_MESSAGE"

// TestLandLeavesNoManualLog holds a landing to needing no file of the
// worker's. Its echo keeps the lines that decide or explain the outcome -- a
// failing test with the detail under it, the verdict, what the gate measured
// -- out of a transcript whose bulk is selection lists and a line per
// package, however the writes are split; the refusal summary keeps the same
// lines; and a message given on standard input reaches the tool run.
func TestLandLeavesNoManualLog(t *testing.T) {
	if os.Getenv(messageEcho) != "" {
		message, _ := io.ReadAll(os.Stdin)
		os.Stdout.WriteString("plan: received " + string(message) + "\n")
		return
	}
	transcript := strings.Join([]string{
		"preflight: selection: complete pending=[overgo/cmd/a,overgo/cmd/b]",
		"preflight: 0 finding(s) in 15 check(s); 0 skipped",
		"gate: phase=test heartbeat=running",
		"gate: step=test package=overgo/cmd/a result=pass elapsed=67ms",
		"gate: step=test package=overgo/cmd/b result=fail elapsed=1.2s",
		"--- FAIL: TestSomething (0.01s)",
		"    thing_test.go:12: got 3, want 4",
		"\tthing_test.go:13: and another",
		"FAIL\tovergo/cmd/b\t1.2s",
		"gate: step=test package=overgo/cmd/c result=pass elapsed=5ms",
		"    an indented line that follows no failure",
		"GATE FAILED 12.0s | ran=test",
		"advisory: class: operation class: code-change",
		"advisory: delta: code profile delta vs HEAD: runtime=+0 files/+29 nodes",
		"advisory: scope: test scope: 194 opaque reader(s) bound to every root",
		"advisory: debt: harness surface raised against paydown row x",
	}, "\n") + "\n"
	want := strings.Join([]string{
		"preflight: 0 finding(s) in 15 check(s); 0 skipped",
		"gate: step=test package=overgo/cmd/b result=fail elapsed=1.2s",
		"--- FAIL: TestSomething (0.01s)",
		"    thing_test.go:12: got 3, want 4",
		"\tthing_test.go:13: and another",
		"FAIL\tovergo/cmd/b\t1.2s",
		"GATE FAILED 12.0s | ran=test",
		"advisory: class: operation class: code-change",
		"advisory: delta: code profile delta vs HEAD: runtime=+0 files/+29 nodes",
		"advisory: debt: harness surface raised against paydown row x",
	}, "\n")
	for _, chunk := range []int{len(transcript), 7, 1} {
		var echoed strings.Builder
		filter := &verdictLines{out: &echoed}
		for rest := transcript; rest != ""; {
			size := min(chunk, len(rest))
			if _, err := filter.Write([]byte(rest[:size])); err != nil {
				t.Fatal(err)
			}
			rest = rest[size:]
		}
		if got := strings.TrimSpace(echoed.String()); got != want {
			t.Fatalf("writes of %d bytes echoed:\n%s\nwant:\n%s", chunk, got, want)
		}
	}
	if got := refusalOf(transcript); got != want {
		t.Fatalf("refusal summary:\n%s\nwant:\n%s", got, want)
	}
	if got := refusalOf("exit status 1\n"); got != "exit status 1" {
		t.Fatalf("a run with nothing telling summarised as %q, want its whole output", got)
	}

	t.Setenv(messageEcho, "1")
	var echoed strings.Builder
	out, err := runToolEnv(os.Environ(), strings.NewReader("Cause: on standard input"), &verdictLines{out: &echoed}, os.Args[0], "-test.run=^TestLandLeavesNoManualLog$")
	if received := "plan: received Cause: on standard input"; err != nil || !strings.Contains(out, received) || !strings.Contains(echoed.String(), received) {
		t.Fatalf("tool run: %v; output %q, echoed %q", err, out, echoed.String())
	}
}
