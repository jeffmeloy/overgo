package main

import (
	"strings"
	"testing"
)

// TestLaneNamesFailedAndUnfinishedTests holds the lane to naming, in its own
// last words, the tests of a failed run that failed and the ones that were
// started and never ended -- the hung test of a timeout, whose name go test
// prints before a goroutine dump longer than anything that keeps only the
// end of the output. The stream passes through unchanged however it is
// split, and subtests, pauses and continuations do not confuse the count.
func TestLaneNamesFailedAndUnfinishedTests(t *testing.T) {
	t.Parallel()
	stream := strings.Join([]string{
		"=== RUN   TestWebUIBrowserPasses",
		"=== PAUSE TestWebUIBrowserPasses",
		"=== RUN   TestWebUIBrowserFails",
		"=== RUN   TestWebUIBrowserFails/leg",
		"    page_test.go:12: got 3, want 4",
		"    --- FAIL: TestWebUIBrowserFails/leg (0.10s)",
		"--- FAIL: TestWebUIBrowserFails (0.12s)",
		"=== CONT  TestWebUIBrowserPasses",
		"--- PASS: TestWebUIBrowserPasses (0.92s)",
		"=== RUN   TestWebUIBrowserSkipped",
		"--- SKIP: TestWebUIBrowserSkipped (0.00s)",
		"=== RUN   TestWebUIBrowserHangs",
		"=== RUN   TestWebUIBrowserHangs/capture",
		"panic: test timed out after 20m0s",
		"\trunning tests:",
		"\t\tTestWebUIBrowserHangs/capture (19m58s)",
		"goroutine 1 [chan receive]:",
	}, "\n") + "\n"
	for _, chunk := range []int{len(stream), 11, 1} {
		var passed strings.Builder
		progress := &testProgress{out: &passed}
		for rest := stream; rest != ""; {
			size := min(chunk, len(rest))
			if _, err := progress.Write([]byte(rest[:size])); err != nil {
				t.Fatal(err)
			}
			rest = rest[size:]
		}
		if passed.String() != stream {
			t.Fatalf("writes of %d bytes changed the stream", chunk)
		}
		if got, want := progress.summary(), "failed=[TestWebUIBrowserFails] unfinished=[TestWebUIBrowserHangs]"; got != want {
			t.Fatalf("writes of %d bytes: summary = %q, want %q", chunk, got, want)
		}
	}
}
