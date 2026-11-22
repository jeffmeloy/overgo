package main

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARemakeRequestNamesOneCaptureAndAFullCommit holds the command's flags to one remake: the CPU ASR reference with
// its election, or a transducer capture with its list, at a full lowercase commit. A request naming neither, both, a
// capture without the record it rewrites, an abbreviated or uppercase commit, or a stray argument is refused before
// anything is read; and a commit the repository's history does not hold is refused before any speech path runs.
func TestARemakeRequestNamesOneCaptureAndAFullCommit(t *testing.T) {
	t.Parallel()
	commit := strings.Repeat("a", 40)
	asr := []string{"-commit", commit, "-asr", "encoder_capture.json", "-election", "election.json"}
	transducer := []string{"-commit", commit, "-transducer", "transducer_captures.json", "-capture", "evidence:sha256:" + strings.Repeat("b", 64)}
	for _, args := range [][]string{asr, transducer} {
		request := testcheck.Must(parse(args))(t)
		if request.commit != commit {
			t.Errorf("%v reads as the commit %q", args, request.commit)
		}
	}
	for name, args := range map[string][]string{
		"neither capture":       {"-commit", commit},
		"both captures":         append(append([]string{}, asr...), transducer[2:]...),
		"no election":           {"-commit", commit, "-asr", "encoder_capture.json"},
		"no transducer capture": {"-commit", commit, "-transducer", "transducer_captures.json"},
		"an abbreviated commit": append([]string{"-commit", commit[:12]}, asr[2:]...),
		"an uppercase commit":   append([]string{"-commit", strings.ToUpper(commit)}, asr[2:]...),
		"no commit":             asr[2:],
		"a stray argument":      append(append([]string{}, asr...), "stray"),
	} {
		if _, err := parse(args); err == nil {
			t.Errorf("a request with %s is admitted", name)
		}
	}
	if err := speechCodeAt(t.Context(), testutil.RepoRoot(t), strings.Repeat("0", 40)); err == nil {
		t.Error("a commit the history does not hold is admitted as the speech path's maker")
	}
}
