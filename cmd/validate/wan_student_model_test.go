//go:build modeltest

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// studentRows reads the store's scorecard rows of the Self-Forcing student and its 50-step base, Wan2.1-T2V-1.3B.
func studentRows(t *testing.T) (ctx context.Context, store *overgodb.Store, student, base scorecard.Row) {
	t.Helper()
	ctx = t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store = overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	capability := mediacapability.Catalog[recipe.TaskVideoGen]
	rows := testcheck.Must(scorecard.BuildOver(ctx, store, discovery.LoadMemo(ctx, store)))(t)
	rowOf := func(name string) scorecard.Row {
		model := testcheck.Must(capability.Resolve(ctx, modeltest.Directory(t, name)))(t).Inventory.Manifest.ID
		at := slices.IndexFunc(rows, func(row scorecard.Row) bool { return row.Model == model && row.Task == recipe.TaskVideoGen })
		if at < 0 {
			t.Fatalf("the scorecard holds no text-to-video row of %s", name)
		}
		return rows[at]
	}
	return ctx, store, rowOf("Self-Forcing"), rowOf("Wan2.1-T2V-1.3B")
}

// TestTheWanStudentHasItsOwnCells holds the Self-Forcing student, a four-step distillation of Wan2.1-T2V-1.3B, to a row
// of its own beside its 50-step base's: wan-few-step/served activated it, and a person's size and held-out quality are
// what a student is for and what it may cost. Its row holds a user-size cell at its own preset, 81 frames at 832 x 480
// in four steps, and both rows a held-out quality on the generated-video suite (TestTheWanStudentIsJudgedBesideItsBase
// compares them).
func TestTheWanStudentHasItsOwnCells(t *testing.T) {
	t.Parallel()
	_, _, studentRow, baseRow := studentRows(t)
	sized := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 4, Guidance: 1, Sampler: "chunk-causal flow", Shift: 5}
	if !slices.ContainsFunc(studentRow.Warm, func(cell scorecard.Cell) bool { return cell.Measured() && cell.Preset == sized }) {
		t.Fatalf("the student's row holds no user-size cell at %+v: %+v", sized, studentRow.Warm)
	}
	for _, cell := range []scorecard.Cell{studentRow.Cells[metric.Quality], baseRow.Cells[metric.Quality]} {
		if !cell.Measured() || !strings.HasPrefix(cell.Measure, generatedVideoSource) {
			t.Fatalf("a quality cell reads %+v, want one of %s", cell, generatedVideoSource)
		}
	}
}

// generatedVideoSource is the suite a text-to-video model's held-out quality is read on.
const generatedVideoSource = "vidgen-1m/generated-video-caption-matching"

// verdictTail is the one-sided tail at which the paired comparison of the student with its base states a difference.
const verdictTail = 0.05

// drawnAt is the size, frames and format a clip was drawn at, as the request that drew it records them: a request that
// names no format drew a GIF.
type drawnAt struct {
	Frames int    `json:"frames"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Format string `json:"format"`
}

// judgedSettings are the sizes and frames of the clips a quality cell's judged run asked about: the run's report names
// its plan, the plan the media it bound, and each medium the request that drew it.
func judgedSettings(t *testing.T, ctx context.Context, store *overgodb.Store, source artifact.ID) []drawnAt {
	t.Helper()
	read := func(id artifact.ID, value any) {
		t.Helper()
		content, found, err := artifact.ReadContent(ctx, store, id)
		if err != nil || !found {
			t.Fatalf("the store holds no %s: %v", id, err)
		}
		testcheck.NoError(t, json.Unmarshal(content.Data, value))
	}
	evaluated := testcheck.Must(runrecord.RequireEvaluation(ctx, store, source))(t)
	run := testcheck.Must(runrecord.RequireRun(ctx, store, evaluated.Run))(t)
	if len(run.Outputs) != 1 {
		t.Fatalf("the run %s behind %s scored no report of its own", run.ID, source)
	}
	var report struct {
		Plan artifact.ID `json:"plan"`
	}
	read(run.Outputs[0], &report)
	var plan struct {
		Media struct {
			Inputs []struct {
				Request artifact.ID `json:"request"`
			} `json:"inputs"`
		} `json:"media"`
	}
	read(report.Plan, &plan)
	var drawn []drawnAt
	for _, input := range plan.Media.Inputs {
		var at drawnAt
		read(input.Request, &at)
		if !slices.Contains(drawn, at) {
			drawn = append(drawn, at)
		}
	}
	return drawn
}

// TestTheWanStudentIsJudgedBesideItsBase holds the comparison of the student with its 50-step base to like with like:
// on 2026-11-08 the student's 0.898 was read beside a base judged at 17 frames, 416 x 240 and 20 steps, and the check
// passed whatever the result. Both quality cells rest on the same cases of the harder generated-video suite, every clip
// either judged drawn at the student's settings, 81 frames at 832 x 480, and in one format on both sides, since a GIF
// differs from an MP4 by its palette as well as its model; the clips are compared one by one, a clip whose questions
// the student answers more of than the base's a win, fewer a loss, and the exact paired test states the verdict at
// verdictTail.
func TestTheWanStudentIsJudgedBesideItsBase(t *testing.T) {
	t.Parallel()
	ctx, store, studentRow, baseRow := studentRows(t)
	want := drawnAt{Frames: 81, Width: 832, Height: 480}
	// atSettings is a row's held-out quality on the generated-video suite judged on clips all drawn at want's size and
	// frames, whichever of its readings heads the row (the base's 128-clip reading at 17 frames, 416 x 240 and 20 steps
	// rests on more cases), and the formats they were drawn in.
	atSettings := func(name string, row scorecard.Row) (scorecard.Cell, []string) {
		var seen []string
		for _, cell := range row.Measured(metric.Quality) {
			if !cell.Measured() || !strings.HasPrefix(cell.Measure, generatedVideoSource) {
				continue
			}
			drawn := judgedSettings(t, ctx, store, cell.Source)
			var formats []string
			for _, clip := range drawn {
				if !slices.Contains(formats, clip.Format) {
					formats = append(formats, clip.Format)
				}
				clip.Format = ""
				if clip != want {
					formats = nil
					break
				}
			}
			if len(formats) > 0 {
				return cell, formats
			}
			seen = append(seen, fmt.Sprintf("%s at %+v", cell.Measure, drawn))
		}
		t.Fatalf("the %s holds no quality judged on %s clips drawn at %+v; its readings: %v", name, generatedVideoSource, want, seen)
		return scorecard.Cell{}, nil
	}
	studentQuality, studentFormats := atSettings("student", studentRow)
	baseQuality, baseFormats := atSettings("base", baseRow)
	if len(studentFormats) != 1 || !slices.Equal(studentFormats, baseFormats) {
		t.Fatalf("the student's clips were drawn as %q and the base's as %q (\"\" a GIF): a comparison holds one format", studentFormats, baseFormats)
	}
	studentCases, read := evaluation.CaseOutcomes(ctx, store, studentQuality.Source)
	baseCases, baseRead := evaluation.CaseOutcomes(ctx, store, baseQuality.Source)
	if !read || !baseRead || !slices.Equal(slices.Sorted(maps.Keys(studentCases)), slices.Sorted(maps.Keys(baseCases))) {
		t.Fatalf("the two cells rest on %d and %d cases that are not the same ones (read %t, %t)", len(studentCases), len(baseCases), read, baseRead)
	}
	right := func(cases map[string]bool) map[string]int {
		clips := map[string]int{}
		for name, answered := range cases {
			clip, count := name[:strings.LastIndex(name, "/")], 0
			if answered {
				count++
			}
			clips[clip] += count
		}
		return clips
	}
	studentClips, baseClips := right(studentCases), right(baseCases)
	var wins, losses, ties uint64
	for clip, held := range baseClips {
		switch answered := studentClips[clip]; {
		case answered > held:
			wins++
		case answered < held:
			losses++
		default:
			ties++
		}
	}
	paired := tensorstats.ExactPairedBinary(wins, losses, ties)
	verdict := "no difference its clips support"
	switch {
	case paired.Tail <= verdictTail:
		verdict = "the student is better"
	case tensorstats.ExactPairedBinary(losses, wins, ties).Tail <= verdictTail:
		verdict = "the student is worse"
	}
	t.Logf("the student %.3f against the base's %.3f on %d cases, both drawn at %d frames, %d x %d, as %q: by clip %d wins, %d losses, %d ties; %s",
		studentQuality.Value, baseQuality.Value, len(studentCases), want.Frames, want.Width, want.Height, studentFormats[0], wins, losses, ties, verdict)
}
