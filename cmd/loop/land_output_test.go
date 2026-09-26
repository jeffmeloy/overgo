package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

const messageEcho = "OVERGO_TEST_ECHO_MESSAGE"

// TestLandReadsStructuredVerdict holds a landing to the tool's own channels,
// with no filter over its text: the echo is the tool's standard output, its
// verdict, never the progress on standard error; a refusal is the final error
// the tool wrote to its error file, or the tail of standard error from a run
// that wrote none; and a message given on standard input reaches the tool.
func TestLandReadsStructuredVerdict(t *testing.T) {
	if mode := os.Getenv(messageEcho); mode != "" {
		message, _ := io.ReadAll(os.Stdin)
		fmt.Println("GATE FAILED | received " + string(message))
		fmt.Fprintln(os.Stderr, "gate: step=test package=overgo/cmd/a result=pass")
		if mode == "typed" {
			clioptions.Main(func() error { return errors.New("gate: test: TestSomething: got 3, want 4") })
		}
		os.Exit(1)
	}
	world := execRowWorld{message: []byte("Cause: on standard input")}
	arguments := []string{os.Args[0], "-test.run=^TestLandReadsStructuredVerdict$"}
	for mode, want := range map[string]string{
		"typed":   "gate: test: TestSomething: got 3, want 4",
		"untyped": "gate: step=test package=overgo/cmd/a result=pass",
	} {
		t.Setenv(messageEcho, mode)
		var echoed strings.Builder
		refusal, err := world.tool(&echoed, arguments...)
		if err == nil || refusal != want {
			t.Fatalf("%s run refused %q (%v), want %q", mode, refusal, err, want)
		}
		if got := echoed.String(); got != "GATE FAILED | received Cause: on standard input\n" {
			t.Fatalf("%s run echoed %q, want the verdict alone", mode, got)
		}
	}
	if refusal, err := world.tool(io.Discard, "go", "version"); err != nil || refusal != "" {
		t.Fatalf("a passing run refused %q (%v)", refusal, err)
	}
}
