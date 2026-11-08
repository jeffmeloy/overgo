package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAMeasuredRequestPrintsWhatItRecorded holds recipe measure to stating the cell it recorded: on 2026-11-08 the
// Self-Forcing student's time at a person's size, about 30 s for 81 frames at 832 x 480 against its base's 374 s, was
// recorded in the store and stated by no commit message or README line, since measure printed its stage ledger and
// the clip and not the cell. Its last line names the observation, its time, the preset the clip ran, its peak device
// memory and that it was a person's size; a request that generates nothing names no preset.
func TestAMeasuredRequestPrintsWhatItRecorded(t *testing.T) {
	t.Parallel()
	preset := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 4, Guidance: 1, Sampler: "chunk-causal flow", Shift: 5}
	observation := testcheck.Must(runrecord.NewServingObservation(runrecord.ServingObservation{
		Model:       testutil.ArtifactID(t, artifact.KindModel, "measured student"),
		Recipe:      testutil.ArtifactID(t, artifact.KindRecipe, "measured recipe"),
		Environment: testutil.ArtifactID(t, artifact.KindEvidence, "measured environment"),
		Task:        recipe.TaskVideoGen, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: uint64(30100 * time.Millisecond),
		UserSize: true, Preset: preset, Resources: runrecord.ServingResources{PeakDeviceBytes: 29 << 30},
		Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: 1}},
	}))(t)
	want := "recipe: recorded " + observation.ID.String() + ": 30.1 s, " + preset.String() + ", peak 29.00 GiB on the card, at a person's size"
	if line := recordedLine(observation); line != want {
		t.Fatalf("measure's last line reads\n%s\nwant\n%s", line, want)
	}
	observation.Preset, observation.UserSize = media.Preset{}, false
	if line := recordedLine(observation); strings.Contains(line, "frames") || strings.Contains(line, "person") {
		t.Fatalf("a request that generated nothing at its own size reads %s", line)
	}
}
