package landing

import (
	"bytes"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// TestCommitChecksRefuseBelowTheFreeSpaceFloor holds a commit's checks to the
// free-space floor. With a reader of free space given, the checks read the
// volume of the commit's own temporary directory once, before any other
// stage: below the floor, or when the read fails, the run is refused as the
// stage space, a preflight, a change to the plan alone and a run that may
// commit alike, and nothing is committed. Below the floor the refusal gives
// the free space, the floor and the command that lists what can be removed; a
// failed read is said as that, with no free figure. The loop classes that
// refusal as a cause outside the change, with a probe. At the floor the run
// lands and the commit's measured line opens with the free space it found. A
// run given no reader, a fixture's, is held to no floor and has no such
// stage. The real reader reads this machine's temporary directory and refuses
// a directory that is not there. Every figure the stand-in reader returns is
// derived from the floor, so the test holds whatever the measured floor is.
func TestCommitChecksRefuseBelowTheFreeSpaceFloor(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.write("a.txt", "changed\n")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	const floor uint64 = freeSpaceFloor
	gibibytes := func(count uint64) string { return fmt.Sprintf("%.1f GiB", float64(count)/gibibyte) }

	// The stand-in reader: what it returns, how often it was asked, and of which directory.
	var free uint64
	var unread error
	var reads int
	var read string
	reader := func(directory string) (uint64, error) {
		reads++
		read = directory
		if info, err := os.Stat(directory); err != nil || !info.IsDir() {
			t.Errorf("the checks read the free space of %s, which is no directory: %v", directory, err)
		}
		return free, unread
	}
	// run runs the gate once and returns its error, its log and the record it left.
	run := func(preflight bool, reader func(string) (uint64, error)) (string, spent, error) {
		t.Helper()
		var log bytes.Buffer
		reads = 0
		_, err := Run(ctx, Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: &log, Preflight: preflight, Free: reader})
		var runs []spent
		if decodeErr := jsonfile.Decode(filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord), &runs); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		return log.String(), runs[len(runs)-1], err
	}
	head, planned := fixture.git("rev-parse", "HEAD"), fixture.read(plan.Path)

	// Half the floor refuses a preflight, before any other stage.
	free = floor / 2
	log, record, err := run(true, reader)
	if err == nil {
		t.Fatalf("a preflight with %d bytes free, half the floor of %d, passed; its log:\n%s", free, floor, log)
	}
	for _, want := range []string{"land: space:", fmt.Sprint(free), fmt.Sprint(floor), gibibytes(free), gibibytes(floor),
		fmt.Sprint(largestCheckWrites), fmt.Sprint(measurementRunWrites), "go run ./cmd/loop -leftovers"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal below the floor is %q, which lacks %q", err, want)
		}
	}
	if strings.Contains(log, "land: format ok") {
		t.Errorf("a stage ran before the floor refused the run:\n%s", log)
	}
	if stages := slices.Collect(maps.Keys(record.Stages)); record.Outcome != spaceClass || !slices.Equal(stages, []string{spaceClass}) {
		t.Errorf("the refused preflight is recorded as %q after the stages %v, want %q after that stage alone", record.Outcome, stages, spaceClass)
	}
	if strings.HasPrefix(read, fixture.repo) {
		t.Errorf("the checks read the free space of %s, in the checkout, want that of the commit's own temporary directory", read)
	}

	// One byte below the floor refuses a run that may commit, and nothing is committed.
	free = floor - 1
	log, _, below := run(false, reader)
	if below == nil || !strings.Contains(below.Error(), "land: space:") {
		t.Fatalf("a landing with %d bytes free, one below the floor of %d, ended as %v, want it refused by the stage space; its log:\n%s", free, floor, below, log)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != head || fixture.read(plan.Path) != planned || reads != 1 {
		t.Errorf("the refused landing left HEAD at %s (was %s), the plan changed %v, after %d reads of free space; want nothing moved and one read", now, head, fixture.read(plan.Path) != planned, reads)
	}

	// A change to the plan alone keeps one stage of the gate's own, and is held to the floor too.
	fixture.write("a.txt", "base\n")
	filed, err := plan.Parse([]byte(planned))
	if err != nil {
		t.Fatal(err)
	}
	filed.Items = append(filed.Items, plan.Item{ID: "later", Title: "later", Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Title: "later", Status: plan.StatusOpen, Verify: "true"}}})
	if err := plan.Save(fixture.planPath(), filed); err != nil {
		t.Fatal(err)
	}
	log, record, err = run(true, reader)
	if stages := slices.Collect(maps.Keys(record.Stages)); err == nil || !strings.Contains(err.Error(), "land: space:") || !slices.Equal(stages, []string{spaceClass}) {
		t.Fatalf("a change to the plan alone, below the floor, ended as %v after the stages %v, want it refused by the stage space alone; its log:\n%s", err, stages, log)
	}
	fixture.write(plan.Path, planned)
	fixture.write("a.txt", "changed\n")

	// A read that fails refuses, with why, and gives no free figure: what the reader returned beside its failure is no reading.
	free, unread = floor, errors.New("the volume is not there")
	if log, _, err := run(false, reader); err == nil || !strings.Contains(err.Error(), "land: space:") || !strings.Contains(err.Error(), unread.Error()) ||
		!strings.Contains(err.Error(), "could not be read") || strings.Contains(err.Error(), "free "+gibibytes(free)) {
		t.Fatalf("a landing whose read of free space failed ended as %v, want it refused by the stage space with the failure, as a read that failed and with no free figure; its log:\n%s", err, log)
	}
	unread = nil

	// A run given no reader is held to no floor: the fixtures of every other test.
	log, record, err = run(true, nil)
	stages := slices.Sorted(maps.Keys(record.Stages))
	if want := []string{"fix", "format", "invariants", "rules", "verify", "vet"}; err != nil || strings.Contains(log, spaceClass) || !slices.Equal(stages, want) {
		t.Fatalf("a preflight with no reader ended as %v after the stages %v, want it passed after %v and its log free of the stage:\n%s", err, stages, want, log)
	}

	// The refusal is a cause outside the change: the loop classes it as one, with a probe.
	if classed := Classify("row/do", "gate: "+below.Error()); classed.Class != spaceClass || classed.External == "" || classed.Probe == "" {
		t.Errorf("the refusal %q is classed %+v, want the class %q with a cause outside the change and a probe", below, classed, spaceClass)
	}

	// Exactly the floor lands, and the measured line opens with what was found.
	free = floor
	log, _, err = run(false, reader)
	if err != nil {
		t.Fatalf("a landing with exactly the floor free was refused: %v\n%s", err, log)
	}
	body := fixture.git("log", "-1", "--format=%B")
	_, measured, _ := strings.Cut(body, "Measured by the gate: ")
	measured, _, _ = strings.Cut(measured, "\n")
	found, later := strings.Index(measured, "free "+gibibytes(free)+", floor "+gibibytes(floor)), strings.Index(measured, ", format ")
	if !strings.HasPrefix(measured, spaceClass+" ") || found < 0 || later < found {
		t.Errorf("the landed commit's measured line is %q, want it to open with the stage %s and the free space and the floor before the next stage", measured, spaceClass)
	}
	if fixture.git("rev-parse", "HEAD") == head {
		t.Error("the landing at the floor reported no error and HEAD did not move")
	}

	// The real reader, on this machine: only that it reads, never how much is free here.
	volume, err := VolumeFree(t.TempDir())
	if err != nil || volume == 0 {
		t.Fatalf("the temporary directory's volume reads as %d bytes free (%v), want a figure above zero", volume, err)
	}
	t.Logf("the temporary directory's volume has %s free (%d bytes)", gibibytes(volume), volume)
	if absent, err := VolumeFree(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Errorf("a directory that is not there reads as %d bytes free, want the read refused", absent)
	}
}
