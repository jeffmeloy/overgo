package inferencesurface

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheSurfaceRecordsSetsAreWrittenFromTheCode holds the surface-move record's set lines to the tool that writes them
// (fail-early/record-written): a record whose result settings lack a name the code holds, and whose catalog line names
// another package, is written back to the code's sets, every other line as it stood; and a record already the code's is
// left untouched.
func TestTheSurfaceRecordsSetsAreWrittenFromTheCode(t *testing.T) {
	t.Parallel()
	root := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	held := string(testcheck.Must(os.ReadFile(filepath.Join(root, SurfaceRecord)))(t))
	want := held
	for key, set := range recordSets() {
		for line := range strings.Lines(held) {
			if body := strings.TrimRight(line, "\r\n"); strings.HasPrefix(body, key+": ") {
				want = strings.Replace(want, body, key+": "+strings.Join(set, " "), 1)
			}
		}
	}
	stale := strings.Replace(want, " OVERGO_DEVICE_BUDGET_GIB", "", 1)
	stale = strings.Replace(stale, "set-capability-catalog: "+capabilityCatalog, "set-capability-catalog: internal/elsewhere", 1)
	if stale == want {
		t.Fatal("the record holds no result-settings line naming OVERGO_DEVICE_BUDGET_GIB to take a name from")
	}
	path := filepath.Join(t.TempDir(), "surface-moves.txt")
	testcheck.NoError(t, os.WriteFile(path, []byte(stale), 0o644))
	testcheck.NoError(t, WriteSurfaceRecord(path))
	if written := string(testcheck.Must(os.ReadFile(path))(t)); written != want {
		t.Errorf("the record written from the code differs from the code's sets with every other line kept:\n%s", written)
	}
	info := testcheck.Must(os.Stat(path))(t)
	testcheck.NoError(t, WriteSurfaceRecord(path))
	if again := testcheck.Must(os.Stat(path))(t); !again.ModTime().Equal(info.ModTime()) {
		t.Error("a record already the code's was written again")
	}
}
