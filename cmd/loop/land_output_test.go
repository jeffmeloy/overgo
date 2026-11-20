package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/landing"
)

const messageEcho = "OVERGO_TEST_ECHO_MESSAGE"

// TestLandReadsStructuredVerdict holds a landing to the tool's own channels,
// with no filter over its text: the echo is the tool's standard output, its
// verdict, never the progress on standard error, which is said apart as it
// comes, a wait for the card's turn among it; a refusal is the final error
// the tool wrote to its error file, or how a run that wrote none ended and
// the tail of its standard error; and a message given on standard input
// reaches the tool.
// The error file does not outlive the run that read it: the temporary
// directory, here one of the test's own, holds nothing after a refused run.
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
	temporary := t.TempDir()
	for _, variable := range []string{"TMP", "TEMP", "TMPDIR"} {
		t.Setenv(variable, temporary)
	}
	progress := "gate: step=test package=overgo/cmd/a result=pass"
	for mode, want := range map[string]string{
		"typed":   "gate: test: TestSomething: got 3, want 4",
		"untyped": filepath.Base(os.Args[0]) + " ended without its refusal (exit status 1); it last wrote: " + progress + "\n" + progress,
	} {
		t.Setenv(messageEcho, mode)
		var echoed, said strings.Builder
		refusal, err := world.tool(&echoed, &said, arguments...)
		if err == nil || refusal != want {
			t.Fatalf("%s run refused %q (%v), want %q", mode, refusal, err, want)
		}
		if got := echoed.String(); got != "GATE FAILED | received Cause: on standard input\n" {
			t.Fatalf("%s run echoed %q, want the verdict alone", mode, got)
		}
		if !strings.HasPrefix(said.String(), "gate: step=test package=overgo/cmd/a result=pass\n") {
			t.Fatalf("%s run said %q of its progress, want its progress line", mode, said.String())
		}
		if left, err := os.ReadDir(temporary); err != nil || len(left) != 0 {
			t.Fatalf("%s run left %v in the temporary directory (%v), want its error file gone once it was read", mode, left, err)
		}
	}
	if refusal, err := world.tool(io.Discard, io.Discard, "go", "version"); err != nil || refusal != "" {
		t.Fatalf("a passing run refused %q (%v)", refusal, err)
	}
}

// endsWithoutResult marks a run of this test binary as a gate that ends without writing its refusal
// (TestAToolThatEndsWithoutAResultNamesItsCause).
const endsWithoutResult = "OVERGO_TEST_ENDS_WITHOUT_RESULT"

// TestAToolThatEndsWithoutAResultNamesItsCause holds the record of a landing whose gate ended without writing its
// refusal to how the gate ended (verdict-steady): the commit attempt for lanes-package-pass/do at 09:31 on 2026-11-19
// ended in its test stage with no error file, and its record kept only its output's first line, a note on files left
// outside the commit. A gate that writes that note and a test stage's progress, then ends with status 3, is recorded
// by its status and the last line it wrote; a tree ended by its bound is recorded by the bound.
func TestAToolThatEndsWithoutAResultNamesItsCause(t *testing.T) {
	const note, progress = "left as it stands, outside the commit, since the step lists none of it: a.go", "gate: step=test package=overgo/cmd/a"
	if os.Getenv(endsWithoutResult) != "" {
		fmt.Fprintln(os.Stderr, note)
		fmt.Fprintln(os.Stderr, progress)
		os.Exit(3)
	}
	t.Setenv(endsWithoutResult, "1")
	world := execRowWorld{read: true}
	refusal, err := world.tool(io.Discard, io.Discard, os.Args[0], "-test.run=^TestAToolThatEndsWithoutAResultNamesItsCause$")
	cause := landing.Classify("row/do", refusal).Cause
	if err == nil || !strings.Contains(cause, "exit status 3") || !strings.Contains(cause, progress) || !strings.Contains(refusal, note) {
		t.Fatalf("a gate that ended with status 3 and no refusal was recorded as %q from %q (%v); want its status and the last line it wrote, its output after", cause, refusal, err)
	}
	bound := errors.Join(toolExit(1), errors.New("processcontrol: the tree reached its memory limit of 20.7 GiB, the machine's commit limit as it started"))
	if cause := landing.Classify("row/do", endedWithout("gate.exe", bound, note+"\n"+progress)).Cause; !strings.Contains(cause, "memory limit of 20.7 GiB") {
		t.Fatalf("a gate its bound ended was recorded as %q; want the bound that ended it", cause)
	}
}
