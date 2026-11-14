package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAPresetIsAdmittedOnlyAtAPersonsSize holds a render's media to the arm they make in a preset's reading
// (judge-calibration/user-size): the size an arm is drawn at is read from the requests drawn, so the VidGen render drawn
// at 17 frames and 416 x 240 is refused as a preset's evidence against a person's 81 at 832 x 480, while a reference and
// a preset drawn at the person's size, in one format, are read; a directory holding another preset's draws is refused.
func TestAPresetIsAdmittedOnlyAtAPersonsSize(t *testing.T) {
	t.Parallel()
	person := map[string]float64{"frames": 81, "width": 832, "height": 480, "steps": 50}
	manifest := func(preset, format string, frames, width, height, steps int) renderManifest {
		draws := map[string]renderedMedia{}
		for at := range 3 {
			request := testcheck.Must(json.Marshal(map[string]any{"prompt": fmt.Sprint("a river ", at), "frames": frames, "width": width, "height": height, "steps": steps}))(t)
			draws[fmt.Sprint("draw-", at)] = renderedMedia{File: fmt.Sprint("clip-", at), Request: request}
		}
		return renderManifest{Task: recipe.TaskVideoGen, Preset: preset, Format: format, Draws: draws}
	}
	right := map[string]bool{"a": true, "b": true, "c": false}
	arm := func(drawn renderManifest) evaluation.PresetArm { return testcheck.Must(drawnArm(drawn))(t) }

	reference, preset := arm(manifest("", "mp4", 81, 832, 480, 50)), arm(manifest("cfg-zero-star", "mp4", 81, 832, 480, 50))
	if verdict, err := evaluation.AdmitPreset(person, reference, preset, right, right); err != nil || !verdict.Admitted {
		t.Errorf("a preset drawn beside its reference at a person's size was not admitted: %+v, %v", verdict, err)
	}
	smaller := arm(manifest("", "", 17, 416, 240, 20))
	if smaller.Format != "gif" {
		t.Errorf("a clip render that names no format reads as %q; want gif, what its requests asked for by naming none", smaller.Format)
	}
	if _, err := evaluation.AdmitPreset(person, smaller, preset, right, right); err == nil {
		t.Error("a reference drawn at 17 frames and 416 x 240 was taken as a preset's evidence against a person's 81 at 832 x 480")
	}
	mixed := manifest("", "mp4", 81, 832, 480, 50)
	mixed.Draws["draw-small"] = renderedMedia{File: "clip-small", Request: testcheck.Must(json.Marshal(map[string]any{"frames": 17}))(t)}
	if _, err := drawnArm(mixed); err == nil {
		t.Error("a render whose draws were drawn at two sizes was read as one arm")
	}
	directory := t.TempDir()
	held := fmt.Sprintf(`{"task": %q, "preset": "guidance-interval", "format": "mp4", "draws": {}}`, recipe.TaskVideoGen)
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, renderManifestFile), []byte(held), 0o644))
	if _, err := alreadyDrawn(directory, renderManifest{Task: recipe.TaskVideoGen, Preset: "cfg-zero-star", Format: "mp4"}); err == nil {
		t.Error("a directory of one preset's draws was taken for another's")
	}
}

// TestAPresetVerdictRefusesArmsOfAnotherRequest holds a preset's reading to one request but the preset (SQA-379):
// -admit-preset given a reference and a preset's arm whose draws differ beyond the preset, a seed, or whose render is
// of another model, refuses before it reads either judged record, and arms of one request but the preset go on to their
// records, which these are not.
func TestAPresetVerdictRefusesArmsOfAnotherRequest(t *testing.T) {
	t.Parallel()
	painter := testutil.ArtifactID(t, artifact.KindModel, "painter")
	render := func(preset string, seed int, model artifact.ID) string {
		t.Helper()
		request := map[string]any{"prompt": "a river at dawn", "seed": seed, "width": 832, "height": 480}
		if preset != "" {
			request["preset"] = preset
		}
		directory := t.TempDir()
		manifest := renderManifest{Model: model, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "painting"), Task: recipe.TaskImageGen, Preset: preset, Draws: map[string]renderedMedia{
			"draw": {File: "picture.png", Request: testcheck.Must(json.Marshal(request))(t),
				Input: testutil.ArtifactID(t, artifact.KindFile, "asked"), Output: testutil.ArtifactID(t, artifact.KindOutput, "drawn")},
		}}
		testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600))
		return directory
	}
	repository := filepath.Join(t.TempDir(), "store")
	testcheck.NoError(t, testcheck.Must(overgodb.Create(repository))(t).Close())
	reading := func(reference, preset string) error {
		return admitPresetReading(t.Context(), repository, t.TempDir(), reference, preset, "unread-reference-record", "unread-preset-record", io.Discard)
	}
	reference := render("", 7, painter)
	for name, arm := range map[string]string{
		"another seed":  render("apg", 8, painter),
		"another model": render("apg", 7, testutil.ArtifactID(t, artifact.KindModel, "sculptor")),
	} {
		if err := reading(reference, arm); err == nil || !strings.Contains(err.Error(), "no one reading") {
			t.Errorf("a preset's arm of %s was read against its reference: %v", name, err)
		}
	}
	if err := reading(reference, render("apg", 7, painter)); err == nil || strings.Contains(err.Error(), "no one reading") {
		t.Errorf("arms of one request but the preset were refused as another's: %v", err)
	}
}
