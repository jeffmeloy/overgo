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

// TestTheSpaceFloorIsTheGreatestLandedFallOfTheWeek holds the space floor to what runs recorded they needed: no figure
// measured by hand (space-record/do), and no one landing's fall alone (space-week/do), where the last landing's let a
// change to the plan set the floor at 0.0 GiB on 2026-11-16. Of five landings, one that landed eight days ago after
// falling 20 GiB, one that landed after falling 3 GiB, one after 5 GiB, a change to the plan that landed after falling
// almost nothing, and a newest one refused after falling 9 GiB, the floor reads 5 GiB, the greatest a landing that
// landed fell in the week, beside the last measurement run's 7 GiB: a preflight with a byte less than their sum free is
// refused at space naming both falls, and one with their sum passes. A run records its own fall: a watch over a write
// of 64 MiB reads at least that fall.
func TestTheSpaceFloorIsTheGreatestLandedFallOfTheWeek(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.write("a.txt", "changed\n")
	recordFalls(t, fixture.repo, 3*gibibyte, 7*gibibyte)
	record := filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord)
	var runs []spent
	testcheck.NoError(t, jsonfile.Decode(record, &runs))
	now := time.Now()
	landed := func(reference string, ago time.Duration, fall uint64) spent {
		return spent{Reference: reference, Started: now.Add(-ago), Ended: now.Add(-ago + time.Minute), Outcome: outcomeLanded, Kind: kindCommit, Fall: fall}
	}
	refused := landed("newest/do", 10*time.Minute, 9*gibibyte)
	refused.Outcome = "test"
	runs = append(runs, landed("aged/do", 8*24*time.Hour, 20*gibibyte), landed("newer/do", 30*time.Minute, 5*gibibyte),
		landed("plan/do", 20*time.Minute, 1<<20), refused)
	testcheck.NoError(t, jsonfile.Write(record, runs, clioptions.PrivateFileMode))
	floor := uint64(12 * gibibyte)

	for _, free := range []uint64{floor - 1, floor} {
		var log bytes.Buffer
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: &log, Preflight: true,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n"),
			Free:    func(string) (uint64, error) { return free, nil }})
		switch below := free < floor; {
		case below && (err == nil || !strings.Contains(err.Error(), "land: space:") || !strings.Contains(err.Error(), "a landing in the week fell at most 5.0 GiB") ||
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
	// The volume is shared: whatever frees space beside the write lowers the fall the watch reads (a landing's parallel
	// tests removing their temporary files took one 4 KiB cluster off it on 2026-11-18, and the volume read about 3 MB
	// short of the write in two landings of 2026-11-20 while the lanes ran beside them), so the watch is held to half
	// the write, which tells a fall it saw from none whatever the volume frees beside it.
	const size, besideSlack = 64 << 20, 32 << 20
	testcheck.NoError(t, os.WriteFile(written, make([]byte, size), 0o600))
	if fall := testcheck.Must(stop())(t); fall < size-besideSlack {
		t.Errorf("a watch over a write of %d bytes read a fall of %d bytes; want the write, less %d bytes freed beside it at most", size, fall, besideSlack)
	}
}
