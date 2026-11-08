//go:build modeltest

package main

import (
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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheWanStudentHasItsOwnCells holds the Self-Forcing student, a four-step distillation of Wan2.1-T2V-1.3B, to a row
// of its own beside its 50-step base's: wan-few-step/served activated it, and a person's size and held-out quality are
// what a student is for and what it may cost. Its row holds a user-size cell at its own preset, 81 frames at 832 x 480
// in four steps, and a held-out quality on the generated-video caption-matching suite the base's quality rests on, the
// same 64 VidGen captions asked twice each. The two are compared clip by clip: a clip the student's clip answers more of
// its two questions about than the base's is a win, fewer a loss, and the exact paired test's tail is stated.
func TestTheWanStudentHasItsOwnCells(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	capability := mediacapability.Catalog[recipe.TaskVideoGen]
	modelOf := func(name string) artifact.ID {
		return testcheck.Must(capability.Resolve(ctx, modeltest.Directory(t, name)))(t).Inventory.Manifest.ID
	}
	student, base := modelOf("Self-Forcing"), modelOf("Wan2.1-T2V-1.3B")
	rows := testcheck.Must(scorecard.BuildOver(ctx, store, discovery.LoadMemo(ctx, store)))(t)
	rowOf := func(model artifact.ID) scorecard.Row {
		at := slices.IndexFunc(rows, func(row scorecard.Row) bool { return row.Model == model && row.Task == recipe.TaskVideoGen })
		if at < 0 {
			t.Fatalf("the scorecard holds no text-to-video row of %s", model)
		}
		return rows[at]
	}
	studentRow, baseRow := rowOf(student), rowOf(base)
	sized := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 4, Guidance: 1, Sampler: "chunk-causal flow", Shift: 5}
	if !slices.ContainsFunc(studentRow.Warm, func(cell scorecard.Cell) bool { return cell.Measured() && cell.Preset == sized }) {
		t.Fatalf("the student's row holds no user-size cell at %+v: %+v", sized, studentRow.Warm)
	}
	studentQuality, baseQuality := studentRow.Cells[metric.Quality], baseRow.Cells[metric.Quality]
	const source = "vidgen-1m/generated-video-caption-matching"
	for _, cell := range []scorecard.Cell{studentQuality, baseQuality} {
		if !cell.Measured() || cell.Cases != 128 || !strings.HasPrefix(cell.Measure, source) {
			t.Fatalf("a quality cell reads %+v, want 128 cases of %s", cell, source)
		}
	}
	studentCases, read := evaluation.CaseOutcomes(ctx, store, studentQuality.Source)
	baseCases, baseRead := evaluation.CaseOutcomes(ctx, store, baseQuality.Source)
	if !read || !baseRead {
		t.Fatalf("the cases of %s or %s cannot be read", studentQuality.Source, baseQuality.Source)
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
		answered, asked := studentClips[clip]
		switch {
		case !asked:
			t.Fatalf("the student was not asked about clip %s, which the base was", clip)
		case answered > held:
			wins++
		case answered < held:
			losses++
		default:
			ties++
		}
	}
	if len(studentClips) != len(baseClips) {
		t.Fatalf("the student was asked about %d clips and the base about %d", len(studentClips), len(baseClips))
	}
	paired := tensorstats.ExactPairedBinary(wins, losses, ties)
	t.Logf("the student %.3f against the base's %.3f on %d cases: by clip %d wins, %d losses, %d ties, tail %.3f",
		studentQuality.Value, baseQuality.Value, studentQuality.Cases, paired.Wins, paired.Losses, paired.Ties, paired.Tail)
}
