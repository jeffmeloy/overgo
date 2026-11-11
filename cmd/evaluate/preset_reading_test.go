package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
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
