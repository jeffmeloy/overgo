package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"
)

// TestPilotTakesNoNumberItWasNotGiven holds the instruction pilot's command
// to stating every number it fixes: asked to fix a pilot with any one of its
// numbers or a part's weight left out, it names what is required and opens no
// store and no model.
func TestPilotTakesNoNumberItWasNotGiven(t *testing.T) {
	stated := []string{
		"-pilot", "base.gguf", "-pool-weights", "development=2,selection=1,promotion=4,audit=1",
		"-pilot-gain", "0.2", "-pilot-seed", "7", "-pilot-window", "2048", "-pilot-tokens", "1280", "-pilot-draws", "8", "-pilot-passes", "1", "-pilot-memory", "24",
	}
	var out bytes.Buffer
	for at := 2; at < len(stated); at += 2 {
		args := slices.Concat([]string{"-repo", t.TempDir()}, stated[:at], stated[at+2:])
		want := "are required"
		if stated[at] == "-pool-weights" {
			want = "a positive weight for each"
		}
		if err := run(args, &out); err == nil || !strings.Contains(err.Error(), want) || out.Len() != 0 {
			t.Errorf("a pilot without %s = %v, having written %q; want it refused for what is missing", stated[at], err, out.String())
		}
	}
}
