//go:build windows

package latentimage

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestASamplingMakesItsControlZerosOnce holds control-zeros/do (SQA-430): a sampling's control feed reads its control
// within the window and zeros outside it, uploading each once, and a step outside the window after the first makes no
// zeros: the feed allocates nothing there. controlAt had made the zeros at every step outside the window before asking
// whether the card held them, 56 buffers of 2.36 MB for one upload in the depth section's pass-2 recording.
func TestASamplingMakesItsControlZerosOnce(t *testing.T) {
	// Serial: testing.AllocsPerRun counts the process's allocations and refuses a parallel test.
	const elements = 64
	control := make([]float32, elements)
	for index := range control {
		control[index] = float32(index) + 1
	}
	var uploads [][]float32
	feed := controlFeed{control: control, low: 0.5, high: 0.9, upload: func(data []float32) (*residentLatent, error) {
		uploads = append(uploads, data)
		return &residentLatent{}, nil
	}}
	read := map[*residentLatent]int{}
	for _, sigma := range []float64{1, 0.95, 0.9, 0.7, 0.5, 0.4, 0.2, 0.1, 0} {
		read[testcheck.Must(feed.at(sigma, elements))(t)]++
	}
	// The schedule opens above the window, so the zeros are uploaded first.
	if len(uploads) != 2 || !slices.Equal(uploads[0], make([]float32, elements)) || !slices.Equal(uploads[1], control) {
		t.Fatalf("the feed uploaded %d slots, %v; want the zeros once, then the control once", len(uploads), uploads)
	}
	if read[feed.slots[0]] != 3 || read[feed.slots[1]] != 6 {
		t.Errorf("the control was read at %d steps and the zeros at %d; want the window's 3 and the other 6", read[feed.slots[0]], read[feed.slots[1]])
	}
	if made := testing.AllocsPerRun(10, func() { _, _ = feed.at(0.1, elements) }); made != 0 {
		t.Errorf("a step outside the window after the first made %.0f allocations; want none", made)
	}
}
