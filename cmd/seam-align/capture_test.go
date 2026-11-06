package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/testcheck"
)

// TestCaptureHoldsTheTrainingSplit holds a capture's samples to the training side of the suite they stand beside: no
// sample is one of the suite's clips or of their source videos, and none is captioned as one of a case's wrong answers;
// the samples are as many as the capture asks, the same for its seed and others for another seed.
func TestCaptureHoldsTheTrainingSplit(t *testing.T) {
	t.Parallel()
	const sources, scenes, cases, samples = 24, 2, 4, 10
	clips := t.TempDir()
	var rows []map[string]string
	for source := range sources {
		for scene := range scenes {
			vid := fmt.Sprintf("source%02d-Scene-%04d", source, scene)
			rows = append(rows, map[string]string{"vid": vid, "caption": "a caption of " + vid})
			testcheck.NoError(t, os.WriteFile(filepath.Join(clips, vid+".mp4"), nil, 0o600))
		}
	}
	captions := filepath.Join(t.TempDir(), "captions.json")
	testcheck.NoError(t, os.WriteFile(captions, testcheck.Must(json.Marshal(rows))(t), 0o600))
	spec := captureSpec{Clips: clips, Captions: captions, Cases: cases, SuiteSeed: 17, Samples: samples, Seed: 3}
	suite := testcheck.Must(evaluation.BuildFrameDescriptionMatching(evaluation.CaptionMatchingSpec{Clips: clips, Captions: captions, Cases: cases, Seed: 17}))(t)
	heldSources, heldCaptions := map[string]bool{}, map[string]bool{}
	for _, held := range suite.Cases {
		vid := held.Name[strings.LastIndex(held.Name, "/")+1:]
		source, _, _ := strings.Cut(vid, "-Scene-")
		heldSources[source] = true
		for _, candidate := range held.Candidates {
			heldCaptions[candidate] = true
		}
	}
	picked := testcheck.Must(captureSamples(spec))(t)
	if len(picked) != samples {
		t.Fatalf("the capture picked %d samples, want %d", len(picked), samples)
	}
	for _, sample := range picked {
		if heldSources[sample.Source] || heldCaptions[sample.Caption] {
			t.Errorf("sample %s stands on the held-out side of the suite", sample.Vid)
		}
	}
	again := testcheck.Must(captureSamples(spec))(t)
	spec.Seed++
	other := testcheck.Must(captureSamples(spec))(t)
	if !slices.Equal(again, picked) || slices.Equal(other, picked) {
		t.Error("the capture's samples do not follow its seed")
	}
}
