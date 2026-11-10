package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestRunCheckSaysToWriteTheMissingTestFirst gives the worker a step whose check names a test no file holds yet, as a
// real step's check names the test the step adds: on 2026-11-09 the local worker spent 81 calls on
// clip-encode/admit-runtime, whose check named such a test, and edited nothing. run_check runs no test there, and its
// answer names the test, its package and the files the step may change, and says to write the test first.
func TestRunCheckSaysToWriteTheMissingTestFirst(t *testing.T) {
	t.Parallel()
	checkout := t.TempDir()
	for name, content := range map[string]string{"go.mod": "module fixture\n\ngo 1.26\n", "p/p.go": "package p\n"} {
		path := filepath.Join(checkout, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	}
	task := replayTask{Verify: "go test ./p -count=1 -run '^TestTheChangeHolds$'", Files: []string{"p"}}
	tools := task.tools(t.Context(), checkout)
	check := tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "run_check" })]
	said := testcheck.Must(check.run(nil))(t)
	for _, want := range []string{"TestTheChangeHolds in ./p", "Write it first", "may change (p)"} {
		if !strings.Contains(said, want) {
			t.Fatalf("run_check answered %q; want it to say %q", said, want)
		}
	}
}
