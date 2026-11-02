package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestAPictureIsHeldOutByItsBytes: a held-out picture that is a copy of a trained one under another name is refused, naming
// both, where the paths differ and it read as unseen. Pictures of other bytes pass, and so does a manifest of one.
func TestAPictureIsHeldOutByItsBytes(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	write := func(name, content string) mossPicture {
		t.Helper()
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return mossPicture{path: path, caption: name}
	}
	trained := []mossPicture{write("fox.png", "a fox"), write("owl.png", "an owl")}
	copied, other := write("renamed.png", "an owl"), write("cat.png", "a cat")
	err := refuseTrainedHeldOut(trained, []mossPicture{other, copied})
	if err == nil || !strings.Contains(err.Error(), "renamed.png") || !strings.Contains(err.Error(), "owl.png") {
		t.Fatalf("a held-out copy = %v, want it refused, naming both", err)
	}
	if err := refuseTrainedHeldOut(trained, []mossPicture{other}); err != nil {
		t.Fatalf("a picture of other bytes = %v", err)
	}
	if err := refuseTrainedHeldOut(trained, []mossPicture{{path: filepath.Join(directory, "absent.png")}}); err == nil {
		t.Fatal("a held-out picture that cannot be read passes")
	}
}

// TestAMOSSRunIsRefusedBeforeItsFirstPicture: what the files alone refuse is refused before a picture is prepared on the
// device. The trained picture here does not decode, so a run that reached its preparation stops on that instead.
func TestAMOSSRunIsRefusedBeforeItsFirstPicture(t *testing.T) {
	t.Parallel()
	root, manifest := mossRouteFixture(t)
	directory := filepath.Dir(manifest)
	write := func(path, content string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	const unreadable = "no picture"
	write(filepath.Join(directory, "picture-0.png"), unreadable)
	write(filepath.Join(directory, "held-0.png"), unreadable)
	held := filepath.Join(directory, "heldout.tsv")
	write(held, "held-0.png\ta red square\n")
	route := func(extra ...string) error {
		scratch := t.TempDir()
		return runRoute(t.Context(), append([]string{
			"moss", "-model", root, "-pictures", manifest, "-size", "32", "-rank", "2", "-alpha", "2", "-steps", "1",
			"-store", t.TempDir(), "-out", filepath.Join(scratch, "checkpoint"),
		}, extra...))
	}
	report := filepath.Join(t.TempDir(), "heldout.json")
	if err := route("-heldout", held, "-report", report); err == nil || !strings.Contains(err.Error(), "is held out and holds the bytes") {
		t.Fatalf("a held-out copy = %v, want it refused before preparation", err)
	}
	scheduler := filepath.Join(root, "scheduler", "scheduler_config.json")
	config, err := os.ReadFile(scheduler)
	if err != nil {
		t.Fatal(err)
	}
	shifted := regexp.MustCompile(`"training_shift":\s*[0-9.]+`).ReplaceAllString(string(config), `"training_shift": 3`)
	if shifted == string(config) {
		t.Fatal("the fixture's scheduler states no training shift")
	}
	write(scheduler, shifted)
	if err := route(); err == nil || !strings.Contains(err.Error(), "training_shift") {
		t.Fatalf("a shifted training grid = %v, want it refused before preparation", err)
	}
}
