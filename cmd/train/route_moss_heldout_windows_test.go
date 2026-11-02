package main

import (
	"os"
	"path/filepath"
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
