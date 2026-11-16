package landing

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// recordFalls writes, into the checkout at root, the record of a landing that landed after falling landed bytes of
// free space and of a measurement run that fell measured, which the space floor reads.
func recordFalls(t *testing.T, root string, landed, measured uint64) {
	t.Helper()
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	started := time.Now().Add(-time.Hour)
	run := spent{Reference: "earlier/do", Started: started, Ended: started.Add(time.Minute), Outcome: outcomeLanded, Kind: kindCommit, Fall: landed}
	testcheck.NoError(t, jsonfile.Write(filepath.Join(root, processlock.StateDirectory, spentRecord), []spent{run}, clioptions.PrivateFileMode))
	measurement := plan.MeasurementRun{Commit: "earlier", Started: started, Ended: started.Add(time.Minute), Fall: measured}
	testcheck.NoError(t, jsonfile.Write(plan.MeasurementRunPath(root), measurement, clioptions.PrivateFileMode))
}

// TestTheSpaceFloorIsReadFromTheLastRecordedFall holds the space floor to what runs recorded they needed
// (space-record/do), where two figures measured by hand on 2026-10-12 and 10-13 held it, and on 2026-11-15
// dispatch-next/do was refused at 34.1 GiB free while a runaway's page file had taken the disk. Of three landings, an
// older one that landed after falling 3 GiB, a newer one that landed after falling 5 GiB and a newest one refused after
// falling 9 GiB, the floor reads the newer landed one's 5 GiB, beside the last measurement run's 7 GiB: a preflight
// with a byte less than their sum free is refused at space naming both falls, and one with their sum passes. A run
// records its own fall: a watch over a write of 64 MiB reads at least that fall.
func TestTheSpaceFloorIsReadFromTheLastRecordedFall(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.write("a.txt", "changed\n")
	recordFalls(t, fixture.repo, 3*gibibyte, 7*gibibyte)
	record := filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord)
	var runs []spent
	testcheck.NoError(t, jsonfile.Decode(record, &runs))
	now := time.Now()
	runs = append(runs,
		spent{Reference: "newer/do", Started: now.Add(-30 * time.Minute), Ended: now.Add(-29 * time.Minute), Outcome: outcomeLanded, Kind: kindCommit, Fall: 5 * gibibyte},
		spent{Reference: "newest/do", Started: now.Add(-10 * time.Minute), Ended: now.Add(-9 * time.Minute), Outcome: "test", Kind: kindCommit, Fall: 9 * gibibyte})
	testcheck.NoError(t, jsonfile.Write(record, runs, clioptions.PrivateFileMode))
	floor := uint64(12 * gibibyte)

	for _, free := range []uint64{floor - 1, floor} {
		var log bytes.Buffer
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: &log, Preflight: true,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n"),
			Free:    func(string) (uint64, error) { return free, nil }})
		switch below := free < floor; {
		case below && (err == nil || !strings.Contains(err.Error(), "land: space:") || !strings.Contains(err.Error(), "the last landing fell 5.0 GiB") ||
			!strings.Contains(err.Error(), "the last measurement run 7.0 GiB")):
			t.Errorf("a preflight with %d bytes free ended as %v; want it refused at space below the floor of the falls recorded, 5 and 7 GiB", free, err)
		case !below && (err != nil || !strings.Contains(log.String(), fmt.Sprintf("floor %.1f GiB", float64(floor)/gibibyte))):
			t.Errorf("a preflight with the floor of %d bytes free ended as %v, saying:\n%s", free, err, log.String())
		}
	}

	directory := t.TempDir()
	_, stop, err := WatchFall(directory, VolumeFree)
	testcheck.NoError(t, err)
	written := filepath.Join(directory, "written")
	const size = 64 << 20
	testcheck.NoError(t, os.WriteFile(written, make([]byte, size), 0o600))
	if fall := testcheck.Must(stop())(t); fall < size {
		t.Errorf("a watch over a write of %d bytes read a fall of %d bytes; want at least the write", size, fall)
	}
}
