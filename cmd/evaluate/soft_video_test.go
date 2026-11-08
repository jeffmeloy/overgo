package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheVideoSuiteScoresProbabilitiesOnHarderCases holds the command's generated-video suite to its harder cases: a
// video model draws, after the picked VidGen captions, one caption composed of counts, colours and places for each
// composedShare of them, as a picture model does, and the suite asks of each clip as evaluation's test of the same name
// holds.
func TestTheVideoSuiteScoresProbabilitiesOnHarderCases(t *testing.T) {
	t.Parallel()
	type row struct {
		Vid     string `json:"vid"`
		Caption string `json:"caption"`
	}
	var rows []row
	for source := range 6 {
		for scene := range 6 {
			vid := fmt.Sprintf("src%02d-Scene-%04d", source, scene)
			rows = append(rows, row{Vid: vid, Caption: "caption of " + vid})
		}
	}
	captions := filepath.Join(t.TempDir(), "captions.json")
	testcheck.NoError(t, os.WriteFile(captions, testcheck.Must(json.Marshal(rows))(t), 0o600))
	items, composed, prompts, err := vidGenCaptions(captions, 8, 17)
	testcheck.NoError(t, err)
	if len(items) != 8 || len(composed) != 8/composedShare || len(prompts) != len(items)+len(composed) {
		t.Fatalf("%d picked and %d composed captions make %d prompts, want 8, %d and their sum", len(items), len(composed), len(prompts), 8/composedShare)
	}
	for index, item := range composed {
		if drawn := prompts[len(items)+index]; drawn.ID != item.ID || drawn.Prompt != item.Caption || !strings.HasPrefix(item.Caption, "From left to right") {
			t.Fatalf("composed caption %d is drawn as %+v, want %q", index, drawn, item.Caption)
		}
	}
}
