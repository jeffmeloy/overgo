package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestPilotKeepsItsRecordBeforeItRemovesItsCheckpoints holds the order of
// what the pilot command does once its arms are trained and judged.
//
// It writes its record to the record file under its output directory as
// well as to its own output, and only once that file is written and synced
// does it remove the checkpoints it made. Each removal is a line of the same
// file, after the record. A checkpoint the store records where it stands is
// not the pilot's to remove and is kept, with a line saying so. A run after
// it adds its record to the file and replaces nothing.
//
// Nothing is removed when a reason to keep the checkpoints was given, which
// the record states; when the reward arm is adopted, which is then the
// reason; or when the record file cannot be written, which fails the
// command. A reason is taken only with the pilot, and a blank one is
// refused, so a flag given with nothing after it never reads as no flag.
//
// What a record kept stays kept. A run after it that gives no reason keeps
// each checkpoint whose newest line in the record file says kept, the new
// line naming the rule it was kept for, and removes only its own: one no
// record kept, and one whose newest line says a removal failed. The run after
// that keeps them under the same rule, not a longer one, also when its output
// directory is spelt another way.
//
// And a run holds its output directory: a second one into it is refused for
// as long as the first goes.
func TestPilotKeepsItsRecordBeforeItRemovesItsCheckpoints(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	base, dataset, out := writeServingFixture(t, root), filepath.Join(root, "dataset.txt"), filepath.Join(root, "pilot")
	if err := os.WriteFile(dataset, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, base, dataset)
	if err != nil {
		t.Fatal(err)
	}
	// Six checkpoints: five nothing records, as the pilot's own are left, and one the store records where it stands.
	arm := trainingworkflow.PilotArm{Name: "reward"}
	const trained, inStore = 6, 5
	for groups := 1; groups <= trained; groups++ {
		request := trainingworkflow.Request{
			Repository: store, Recipe: recipeID, ModelDirectory: base, DatasetPath: dataset,
			OutputDirectory: filepath.Join(out, arm.Name, fmt.Sprintf("groups-%03d", groups)), Steps: groups, MaximumSequence: 16, Host: true,
		}
		if groups == inStore {
			request.Observations = store
		}
		if _, err := trainingworkflow.Execute(ctx, request); err != nil {
			t.Fatal(err)
		}
		arm.Checkpoints = append(arm.Checkpoints, trainingworkflow.PilotCheckpoint{Groups: groups, Directory: request.OutputDirectory})
	}
	// Two a run keeps for a reason, one an adopted run keeps, one no record keeps, the store's, and one a run fails to remove.
	reasoned, taken, own, recorded, busy := arm.Checkpoints[:2], arm.Checkpoints[2], arm.Checkpoints[3], arm.Checkpoints[4], arm.Checkpoints[5]
	run := func(checkpoints ...trainingworkflow.PilotCheckpoint) trainingworkflow.PilotResult {
		return trainingworkflow.PilotResult{
			Selection: 4, BaseSelected: 1, Reason: `comparison "reward over base" answered loss`,
			Arms: []trainingworkflow.PilotArm{{Name: arm.Name, Checkpoints: checkpoints}},
		}
	}
	result := run(arm.Checkpoints...)
	acceptance := testutil.ArtifactID(t, artifact.KindEvidence, "pilot record fixture acceptance")
	standing := func(when string) {
		t.Helper()
		for _, checkpoint := range arm.Checkpoints {
			if _, err := trainingprogram.LoadCheckpoint(checkpoint.Directory); err != nil {
				t.Fatalf("%s, the checkpoint at %s loads as %v, want it whole", when, checkpoint.Directory, err)
			}
		}
	}
	file := filepath.Join(out, pilotRecord)
	var printed bytes.Buffer
	written := func(when string) string {
		t.Helper()
		content, err := os.ReadFile(file)
		if err != nil || string(content) != printed.String() {
			t.Fatalf("%s, the record file holds %q (%v), want exactly what was printed so far: %q", when, content, err, printed.String())
		}
		return string(content)
	}

	// A reason is the rest of its line, so one that holds a tab is still read whole by the run after.
	const reason = "the second arm's checkpoints\tare to be read by hand"
	if err := recordPilot(ctx, store, acceptance, "fixture commit", out, reason, run(reasoned...), &printed); err != nil {
		t.Fatal(err)
	}
	standing("given a reason to keep")
	if record := written("given a reason to keep"); strings.Count(record, "kept\t") != len(reasoned) || strings.Count(record, reason) != len(reasoned) || strings.Contains(record, "removed\t") {
		t.Fatalf("given a reason to keep, the record is %q, want each checkpoint kept for %q", record, reason)
	}

	adopted := run(taken)
	adopted.Adopted, adopted.Reason = true, ""
	before := printed.Len()
	if err := recordPilot(ctx, store, acceptance, "fixture commit", out, "", adopted, &printed); err != nil {
		t.Fatal(err)
	}
	standing("with the reward arm adopted")
	if record := written("with the reward arm adopted")[before:]; strings.Count(record, "kept\t") != 1 || strings.Count(record, adoptedKept) != 1 || strings.Contains(record, "removed\t") || !strings.Contains(record, "adopted: true") {
		t.Fatalf("with the reward arm adopted, the run's record is %q, want its checkpoint kept because %s", record, adoptedKept)
	}

	// A record file that cannot be read or written: the command fails and removes nothing.
	blocked := filepath.Join(root, "blocked")
	if err := os.MkdirAll(filepath.Join(blocked, pilotRecord), 0o700); err != nil {
		t.Fatal(err)
	}
	var unwritten bytes.Buffer
	if err := recordPilot(ctx, store, acceptance, "fixture commit", blocked, "", result, &unwritten); err == nil || strings.Contains(unwritten.String(), "removed\t") {
		t.Fatalf("with a record file that cannot be written the command ended with %v and printed %q, want it failed with nothing removed", err, unwritten.String())
	}
	standing("with a record file that cannot be written")
	// So is a record that cannot be printed: the run after it finds the file as the run before left it.
	if err := recordPilot(ctx, store, acceptance, "fixture commit", out, "", result, unwritable{}); !errors.Is(err, errUnwritable) {
		t.Fatalf("with an output that cannot be written the command ended with %v, want it failed for that", err)
	}
	standing("with an output that cannot be written")
	written("with an output that cannot be written")

	if runtime.GOOS == "windows" {
		// A checkpoint with a file held open is not removed: the run fails, and its record says the checkpoint was left.
		open, err := os.Open(filepath.Join(busy.Directory, trainingprogram.CheckpointFilename))
		if err != nil {
			t.Fatal(err)
		}
		before = printed.Len()
		err = recordPilot(ctx, store, acceptance, "fixture commit", out, "", run(busy), &printed)
		if closeErr := open.Close(); err == nil || closeErr != nil {
			t.Fatalf("a run whose checkpoint has a file held open ended with %v (close: %v), want an error", err, closeErr)
		}
		standing("with a file of a checkpoint held open")
		if record := written("with a file of a checkpoint held open")[before:]; !strings.Contains(record, "left\t") || !strings.Contains(record, "\t"+busy.Directory+"\t") || strings.Contains(record, "removed\t") {
			t.Fatalf("with a file of the checkpoint at %s held open the run's record is %q, want a line saying it was left", busy.Directory, record)
		}
	}

	// The run that removes, with no reason and nothing adopted, over every checkpoint.
	before = printed.Len()
	if err := recordPilot(ctx, store, acceptance, "fixture commit", out, "", result, &printed); err != nil {
		t.Fatal(err)
	}
	record, lines := pilotReceipt(t, written("after the run that removes")[before:])
	if !strings.HasSuffix(record, "adopted: false "+result.Reason+"\n") {
		t.Fatalf("the run's record is %q, want the record whole, ending with the adoption line, before the first line of what was done with a checkpoint", record)
	}
	// What an earlier record kept, by the rule it was kept for.
	earlier := map[string]string{reasoned[0].Directory: reason, reasoned[1].Directory: reason, taken.Directory: adoptedKept}
	for directory, rule := range earlier {
		line := lines[directory]
		if _, err := trainingprogram.LoadCheckpoint(directory); err != nil || line == nil || line[0] != "kept" || !strings.HasSuffix(line[4], rule) || line[4] == rule {
			t.Fatalf("the checkpoint at %s, which an earlier record kept for %q, loads as %v with the line %q, want it whole and kept again, the line naming that rule and saying what is to remove it", directory, rule, err, line)
		}
		earlier[directory] = line[4]
	}
	// One no record kept, and one whose last line says a run left it: both are the pilot's own to remove.
	for _, checkpoint := range []trainingworkflow.PilotCheckpoint{own, busy} {
		line := lines[checkpoint.Directory]
		if _, err := os.Stat(checkpoint.Directory); !errors.Is(err, os.ErrNotExist) || line == nil || line[0] != "removed" || line[1] == "0" {
			t.Fatalf("the pilot's own checkpoint at %s, which no earlier record kept, is %v with the line %q, want it removed in a line after the record", checkpoint.Directory, err, line)
		}
	}
	line := lines[recorded.Directory]
	if _, err := trainingprogram.LoadCheckpoint(recorded.Directory); err != nil || line == nil || line[0] != "kept" || line[1] != "0" || !strings.Contains(line[4], "the store records this checkpoint") || strings.Count(printed.String()[before:], "removed\t") != 2 {
		t.Fatalf("the checkpoint the store records at %s loads as %v with the line %q, want it whole and kept for the store's record, and two others removed", recorded.Directory, err, line)
	}
	earlier[recorded.Directory] = line[4]
	for _, want := range []string{"fixture commit", acceptance.String(), "base: passed 1 of 4 selection groups", "reward after 2 groups", "reward: frozen after 1 groups, " + reasoned[0].Directory} {
		if !strings.Contains(record, want) {
			t.Fatalf("the run's record %q lacks %q", record, want)
		}
	}

	// The run after it, its output spelt another way: each checkpoint a record kept is kept once more
	// under the rule it was first kept for, and that rule is not wrapped a second time.
	respelt := run()
	spelt := func(directory string) string {
		return filepath.Dir(directory) + string(filepath.Separator) + "." + string(filepath.Separator) + filepath.Base(directory)
	}
	for _, checkpoint := range arm.Checkpoints {
		checkpoint.Directory = spelt(checkpoint.Directory)
		respelt.Arms[0].Checkpoints = append(respelt.Arms[0].Checkpoints, checkpoint)
	}
	before = printed.Len()
	if err := recordPilot(ctx, store, acceptance, "fixture commit", out, "", respelt, &printed); err != nil {
		t.Fatal(err)
	}
	_, lines = pilotReceipt(t, written("after the run that follows the removal")[before:])
	for directory, rule := range earlier {
		line := lines[spelt(directory)]
		if _, err := trainingprogram.LoadCheckpoint(directory); err != nil || line == nil || line[0] != "kept" || !strings.HasSuffix(line[4], rule) || directory != recorded.Directory && line[4] != rule {
			t.Fatalf("the checkpoint at %s, which the record before kept under %q, loads as %v with the line %q, want it whole and kept under that rule", directory, rule, err, line)
		}
	}
	if len(lines) != len(earlier) {
		t.Fatalf("the run after the removal wrote the lines %q, want one for each of the %d checkpoints a record kept and none for a checkpoint that is gone", lines, len(earlier))
	}
	// The newest line for a path decides. No run writes a removal after a keep, so that order is read here from a record made by hand.
	older := "kept\t0\tplace\t" + own.Directory + "\tan older rule\n"
	if rule, kept := keptBefore(older+"removed\t1\tplace\t"+own.Directory+"\ta newer rule\n", own.Directory); kept {
		t.Fatalf("a path whose newest line says removed, after an older one that says kept, reads as kept under %q, want it not kept", rule)
	}
	if rule, kept := keptBefore(older, own.Directory); !kept || rule != "an older rule" {
		t.Fatalf("a path whose one line says kept reads as kept %t under %q, want it kept under its rule", kept, rule)
	}

	// A reason is the pilot's, and is never blank.
	for name, refused := range map[string]error{
		"a reason without the pilot": keepReason(true, "", reason),
		"a blank reason":             keepReason(true, acceptance.String(), " "),
	} {
		if refused == nil {
			t.Errorf("%s was taken, want it refused", name)
		}
	}
	if err := errors.Join(keepReason(true, acceptance.String(), reason), keepReason(false, "", ""), keepReason(false, acceptance.String(), "")); err != nil {
		t.Fatalf("a reason given with the pilot, or none given = %v, want it taken", err)
	}

	// A run holds its output directory.
	held, err := holdOutput(filepath.Join(root, "held"))
	if err != nil {
		t.Fatal(err)
	}
	if second, err := holdOutput(filepath.Join(root, "held")); !errors.Is(err, processlock.ErrBusy) {
		t.Fatalf("a second run into a held output directory = (%v, %v), want it refused as busy", second, err)
	}
	if err := held.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := holdOutput(filepath.Join(root, "held"))
	if err != nil {
		t.Fatalf("a run into an output directory its holder released = %v", err)
	}
	if err := after.Close(); err != nil {
		t.Fatal(err)
	}
}

// errUnwritable is what a write to unwritable returns.
var errUnwritable = errors.New("the output cannot be written")

// unwritable is an output no record can be printed to.
type unwritable struct{}

func (unwritable) Write([]byte) (int, error) { return 0, errUnwritable }

// pilotReceipt splits what one run printed into its record and the lines after
// it that say what was done with a checkpoint or a folder, by path: verb,
// bytes, place, path and rule between tabs, the rule the rest of its line. A
// line of the record after such a line, or a second line for a path, fails
// the test: the record is whole before anything is said to be removed.
func pilotReceipt(t *testing.T, printed string) (record string, lines map[string][]string) {
	t.Helper()
	lines = map[string][]string{}
	for line := range strings.Lines(printed) {
		fields := strings.SplitN(strings.TrimSuffix(line, "\n"), "\t", 5)
		switch {
		case len(fields) == 5 && lines[fields[3]] == nil:
			lines[fields[3]] = fields
		case len(fields) == 5 || len(lines) != 0:
			t.Fatalf("the line %q is its path's second, or a line of the record after a line for a path: want the record whole, and then one line a path", line)
		default:
			record += line
		}
	}
	return record, lines
}
