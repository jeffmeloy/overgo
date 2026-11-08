package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheReadmeTableIsWrittenFromTheScorecard holds the README's speed table to the newest scorecard record beside the
// store: with no record -readme refuses; a hand-typed table fails -check; -readme writes the newest record's table
// between the two marker lines and leaves the rest of the README as it was, after which -check passes; a figure typed
// into the written table fails -check again; a README without the markers is refused, and -check alone is refused.
func TestTheReadmeTableIsWrittenFromTheScorecard(t *testing.T) {
	t.Parallel()
	root, store := t.TempDir(), filepath.Join(t.TempDir(), "overgodb-store")
	records := scorecard.Directory(store)
	readme := filepath.Join(root, readmeFile)
	run := func(args ...string) error {
		return runArgs(t.Context(), append([]string{"-root", root, "-repo", store}, args...), io.Discard)
	}
	opening, closing := "# overgo\n\n### Speed\n\n", "\nStrata reads the same prompts.\n"
	testcheck.NoError(t, os.WriteFile(readme, []byte(opening+readmeSpeedBegin+"\n| hand | 67 tok/s |\n"+readmeSpeedEnd+closing), readmeMode))
	if err := run("-readme"); err == nil || !strings.Contains(err.Error(), "no scorecard record") {
		t.Fatalf("-readme with no record = %v, want it refused", err)
	}
	row := scorecard.Row{Name: "served", Task: recipe.TaskInference, Cells: map[metric.Name]scorecard.Cell{}}
	for _, column := range metric.Columns {
		row.Cells[column] = scorecard.Cell{Reason: "a fixture measures nothing"}
	}
	published := time.Date(2026, time.November, 7, 12, 0, 0, 0, time.UTC)
	for _, offset := range []time.Duration{0, time.Hour} {
		testcheck.Must(scorecard.Publish(records, published.Add(offset), scorecard.Record{Rows: []scorecard.Row{row}}))(t)
	}
	if err := run("-readme", "-check"); err == nil {
		t.Fatal("-readme -check passed a hand-typed speed table")
	}
	testcheck.NoError(t, run("-readme"))
	all := testcheck.Must(scorecard.Records(records))(t)
	want := opening + readmeSpeedBegin + "\n" + all[len(all)-1].SpeedTable() + readmeSpeedEnd + closing
	if got := string(testcheck.Must(os.ReadFile(readme))(t)); got != want {
		t.Fatalf("the README after -readme =\n%s\nwant\n%s", got, want)
	}
	testcheck.NoError(t, run("-readme", "-check"))
	typed := strings.Replace(want, readmeSpeedEnd, "| served | 2,540 tok/s (1.61 s) |\n"+readmeSpeedEnd, 1)
	testcheck.NoError(t, os.WriteFile(readme, []byte(typed), readmeMode))
	if err := run("-readme", "-check"); err == nil || !strings.Contains(err.Error(), all[len(all)-1].Name) {
		t.Fatalf("-readme -check on a typed figure = %v, want it refused naming the newest record", err)
	}
	testcheck.NoError(t, os.WriteFile(readme, []byte(opening+closing), readmeMode))
	if err := run("-readme"); err == nil || !strings.Contains(err.Error(), "holds no speed table") {
		t.Fatalf("-readme on a README without its markers = %v, want it refused", err)
	}
	if err := run("-check"); err == nil {
		t.Fatal("-check alone was admitted")
	}
}

// TestAMediaCellRecordsThePresetItRan holds the README's generated pictures and clips to the newest record: -readme
// writes, after the speed table, each image or video row's time with the preset its attempts ran, so no hand types a
// size or a step count beside a time.
func TestAMediaCellRecordsThePresetItRan(t *testing.T) {
	t.Parallel()
	root, store := t.TempDir(), filepath.Join(t.TempDir(), "overgodb-store")
	readme := filepath.Join(root, readmeFile)
	testcheck.NoError(t, os.WriteFile(readme, []byte(readmeSpeedBegin+"\n"+readmeSpeedEnd+"\n"), readmeMode))
	row := scorecard.Row{Name: "Wan", Task: recipe.TaskVideoGen, Cells: map[metric.Name]scorecard.Cell{}}
	for _, column := range metric.Columns {
		row.Cells[column] = scorecard.Cell{Reason: "a fixture measures nothing"}
	}
	row.Cells[metric.FirstOutput] = scorecard.Cell{Value: 13_000, Unit: "ms", Source: testutil.ArtifactID(t, artifact.KindEvidence, "clip attempt"),
		Preset: media.Preset{Width: 32, Height: 16, Frames: 21, Steps: 2, Guidance: 5, Sampler: "flow UniPC", Shift: 5}}
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), time.Date(2026, time.November, 8, 12, 0, 0, 0, time.UTC), scorecard.Record{Rows: []scorecard.Row{row}}))(t)
	testcheck.NoError(t, runArgs(t.Context(), []string{"-root", root, "-repo", store, "-readme"}, io.Discard))
	want := "| Wan | 13.0 s (32 x 16, 21 frames, 2 steps, flow UniPC, guidance 5, shift 5) | — | — |\n" + readmeSpeedEnd
	if written := string(testcheck.Must(os.ReadFile(readme))(t)); !strings.Contains(written, want) {
		t.Fatalf("the README after -readme =\n%s\nwant the clip's time with its preset:\n%s", written, want)
	}
}
