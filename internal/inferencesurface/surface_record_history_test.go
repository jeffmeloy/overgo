package inferencesurface

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// surfaceRecordWritten is the commit that wrote the surface-move record from its probe over 500 commits.
const surfaceRecordWritten = "68f549f6c"

// TestTheSurfaceRecordKeepsItsProbesSets holds the October surface-move record to the sets its probe recorded (SQA-377):
// it is evidence of its own revision, so a later change to the code's sets is no edit of it. Three landings edited its
// result-settings line to today's names (12d049031, f9aeadd1a, a0cfebe16) without changing the probe it names, which
// made it claim what its probe never read.
func TestTheSurfaceRecordKeepsItsProbesSets(t *testing.T) {
	t.Parallel()
	root := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	record := filepath.Join("docs", "verification", "surface-moves.txt")
	show := exec.CommandContext(t.Context(), "git", "show", surfaceRecordWritten+":"+filepath.ToSlash(record))
	show.Dir = root
	written := testcheck.Must(show.Output())(t)
	sets := func(text string) []string {
		var lines []string
		for line := range strings.Lines(text) {
			if line = strings.TrimRight(line, "\r\n"); strings.HasPrefix(line, "set-") {
				lines = append(lines, line)
			}
		}
		return lines
	}
	held := sets(string(testcheck.Must(os.ReadFile(filepath.Join(root, record)))(t)))
	if want := sets(string(written)); !slices.Equal(held, want) {
		t.Errorf("the record's set lines are\n%s\nwant the probe's, as %s wrote them:\n%s", strings.Join(held, "\n"), surfaceRecordWritten, strings.Join(want, "\n"))
	}
}
