package landing

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestTheFirstPassRateCountsAttemptsToEachLanding holds the day's line to how often a commit lands as first written
// (first-pass/do): the run records held every attempt (5 of 11 commits landed on their first on 2026-11-14), and no line
// said so. A landing's attempts are its reference's refused commit attempts since it last landed, and itself; a
// preflight, and a refused run of a record from before runs said their kind, is no attempt. Over the window and since
// the first run that said its kind the line says the landings made on a first attempt, and the most attempts one
// landing took: a whole record counted from Oct 18 read 1115 of 1180, where since 2026-11-11, when runs began to say
// their kind, it is 70 of 135 (first-pass/since).
func TestTheFirstPassRateCountsAttemptsToEachLanding(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	now := time.Now()
	day, before := now.Add(-time.Hour), now.Add(-48*time.Hour)
	run := func(reference string, at time.Time, outcome, kind string) spent {
		return spent{Reference: reference, Started: at, Ended: at.Add(time.Minute), Outcome: outcome, Kind: kind}
	}
	runs := []spent{
		run("older/do", before.Add(-time.Hour), "test", ""), run("older/do", before.Add(-time.Minute), outcomeLanded, ""),
		run("old/do", before, "test", kindCommit), run("old/do", before.Add(time.Minute), outcomeLanded, kindCommit),
		run("a/do", day, "rules", kindCommit), run("a/do", day.Add(time.Minute), "verify", kindCommit),
		run("a/do", day.Add(2*time.Minute), outcomeLanded, kindCommit),
		run("b/do", day.Add(3*time.Minute), "test", ""), run("b/do", day.Add(4*time.Minute), outcomeLanded, kindCommit),
		run("c/do", day.Add(5*time.Minute), "form", kindPreflight), run("c/do", day.Add(6*time.Minute), outcomeLanded, kindCommit),
	}
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	testcheck.NoError(t, jsonfile.Write(filepath.Join(root, processlock.StateDirectory, spentRecord), runs, clioptions.PrivateFileMode))
	delivery := Delivery{Since: now.Add(-deliveryWindow), Refused: map[string]int{}, UnknownKind: map[string]int{}}
	testcheck.NoError(t, delivery.readRuns(root))
	if delivery.FirstPass != 2 || delivery.Landings != 3 || delivery.FirstPassAll != 2 || delivery.LandingsAll != 4 ||
		delivery.Worst != 3 || delivery.WorstReference != "a/do" || !delivery.RecordSince.Equal(before) {
		t.Fatalf("the record reads first pass %d of %d, %d of %d since %v, the most %d (%s); want 2 of 3, 2 of 4 since %v, 3 (a/do)",
			delivery.FirstPass, delivery.Landings, delivery.FirstPassAll, delivery.LandingsAll, delivery.RecordSince, delivery.Worst, delivery.WorstReference, before)
	}
	if line := delivery.String(); !strings.Contains(line, ": first pass 2 of 3 landings, the most attempts 3 (a/do); 2 of 4 since ") {
		t.Fatalf("the day's line %q does not lead with the first-pass rate", line)
	}
}
