package main

import (
	"slices"
	"strconv"
	"testing"

	"overgo/internal/evaluation"
)

// TestHeldSourcesStayOutOfTraining holds the caption projector's held-back share to whole source videos: every sample
// of the first held sources in the capture's order is scored and none is trained on, a later sample of a held source
// with them, and every other sample is trained on in its order; holding none trains on every sample.
func TestHeldSourcesStayOutOfTraining(t *testing.T) {
	t.Parallel()
	sources := []string{"a", "b", "a", "c", "b", "d"}
	samples, sequences := make([]evaluation.CapturedCaption, len(sources)), make([]captionSequence, len(sources))
	for index, source := range sources {
		samples[index].Source, sequences[index].vid = source, strconv.Itoa(index)
	}
	vids := func(chosen []captionSequence) []string {
		names := make([]string, len(chosen))
		for index, sequence := range chosen {
			names[index] = sequence.vid
		}
		return names
	}
	trained, scored := heldBack(samples, sequences, 2)
	if !slices.Equal(vids(scored), []string{"0", "1", "2", "4"}) || !slices.Equal(vids(trained), []string{"3", "5"}) {
		t.Fatalf("holding two sources scored %v and trained %v; want every sample of a and b scored, the rest trained", vids(scored), vids(trained))
	}
	if trained, scored := heldBack(samples, sequences, 0); len(scored) != 0 || len(trained) != len(sources) {
		t.Fatalf("holding no source scored %d samples and trained %d; want every sample trained", len(scored), len(trained))
	}
}
