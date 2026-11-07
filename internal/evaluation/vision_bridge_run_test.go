package evaluation

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"slices"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// visionBridgeRun and visionBridgeNineBRun are the records of the 0.5B and the 9B composites' training, and
// secondsPerHour what their segments' walls are divided by for their card-hours.
const (
	visionBridgeRun      = "../../docs/verification/vision-bridge-0.5b-run.json"
	visionBridgeNineBRun = "../../docs/verification/vision-bridge-9b-run.json"
	secondsPerHour       = 3600
)

// visionBridgeSeam is one seam the run's capture ranked, as seam-align -rank prints it.
type visionBridgeSeam struct {
	Source    string  `json:"source"`
	Width     int     `json:"width"`
	Samples   int     `json:"samples"`
	InSample  float64 `json:"in_sample"`
	HeldOut   float64 `json:"held_out"`
	Unrelated float64 `json:"unrelated"`
	Rung      string  `json:"rung"`
}

// visionBridgeSegment is one stretch of the run's steps: the checkpoint it trained from, its steps and batch, the
// checkpoint it published, the held-back loss before and after, its wall and its peak card memory.
type visionBridgeSegment struct {
	From            string  `json:"from"`
	Steps           int     `json:"steps"`
	Batch           int     `json:"batch"`
	Checkpoint      string  `json:"checkpoint"`
	HeldLossBefore  float64 `json:"held_loss_before"`
	HeldLossAfter   float64 `json:"held_loss_after"`
	WallSeconds     float64 `json:"wall_seconds"`
	PeakDeviceBytes uint64  `json:"peak_device_bytes"`
}

// visionBridgeRunRecord is what the run records: the terms it trained under by their digest, the comparison it serves,
// the capture it trained on and the samples it held back, the seams that capture ranked before any step, each
// training segment in order, those whose held-back loss did not fall and which therefore published nothing, the
// checkpoint the held-back samples chose, and the card-hours the run took.
type visionBridgeRunRecord struct {
	Schema     string `json:"schema"`
	Acceptance string `json:"acceptance"`
	Comparison string `json:"comparison"`
	Capture    struct {
		File        string `json:"file"`
		Samples     int    `json:"samples"`
		HeldSources int    `json:"held_sources"`
		HeldSamples int    `json:"held_samples"`
	} `json:"capture"`
	Ranking  []visionBridgeSeam    `json:"ranking"`
	Segments []visionBridgeSegment `json:"segments"`
	Stopped  []visionBridgeSegment `json:"stopped"`
	Chosen   string                `json:"chosen"`
	GPUHours float64               `json:"gpu_hours"`
}

// TestVisionBridgeRunHoldsItsTerms holds the 0.5B composite's training to the acceptance, its first segment from the
// zero-output bridge, within the 0.5B's card-hours.
func TestVisionBridgeRunHoldsItsTerms(t *testing.T) {
	t.Parallel()
	runHoldsItsTerms(t, visionBridgeRun, "zero", 0, func(terms visionBridgeTerms) float64 { return terms.Limits.GPUHours05B })
}

// TestTheNineBBridgeRunHoldsItsTerms holds the 9B composite's training to the acceptance, its first segment from the
// captions' mean, within the 9B's card-hours.
func TestTheNineBBridgeRunHoldsItsTerms(t *testing.T) {
	t.Parallel()
	runHoldsItsTerms(t, visionBridgeNineBRun, "mean", 1, func(terms visionBridgeTerms) float64 { return terms.Limits.GPUHours9B })
}

// runHoldsItsTerms holds a composite's training record at path to the acceptance: it names the terms by their digest
// and the comparison it serves, holds samples back from training, records its capture's seams ranked best first, trains
// in segments each from the checkpoint before it, the first from start, chooses the checkpoint whose held-back loss is
// least and below its start's, and takes no more card-hours than the terms allow it.
func runHoldsItsTerms(t *testing.T, path, start string, comparison int, hours func(visionBridgeTerms) float64) {
	t.Helper()
	data := testcheck.Must(os.ReadFile(visionBridgeAcceptance))(t)
	var terms visionBridgeTerms
	testcheck.NoError(t, strictjson.DecodeBytes(data, &terms))
	digest := sha256.Sum256(data)
	var run visionBridgeRunRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(path))(t), &run))
	if run.Schema != "overgo/vision-bridge-run/v1" || run.Acceptance != "sha256:"+hex.EncodeToString(digest[:]) || run.Comparison != terms.Comparisons[comparison].Name {
		t.Fatalf("the run names the terms %s and the comparison %q; want the acceptance's digest and its comparison %q", run.Acceptance, run.Comparison, terms.Comparisons[comparison].Name)
	}
	if run.Capture.File == "" || run.Capture.HeldSources <= 0 || run.Capture.HeldSamples <= 0 || run.Capture.HeldSamples >= run.Capture.Samples {
		t.Errorf("the run holds back %d samples of %d sources from %d; want a share held back to choose its checkpoint",
			run.Capture.HeldSamples, run.Capture.HeldSources, run.Capture.Samples)
	}
	byHeldOut := func(left, right visionBridgeSeam) int { return cmp.Compare(left.HeldOut, right.HeldOut) }
	if len(run.Ranking) == 0 || !slices.IsSortedFunc(run.Ranking, byHeldOut) || run.Ranking[0].Samples != run.Capture.Samples {
		t.Error("the run records no ranking of its capture's seams, best first")
	}
	if len(run.Segments) == 0 || run.Segments[0].From != start {
		t.Fatalf("the run records no segment trained from its start, %s", start)
	}
	chosen, seconds := 0, 0.0
	for index, segment := range run.Segments {
		if index > 0 && segment.From != run.Segments[index-1].Checkpoint {
			t.Errorf("segment %d trains from %s, not the checkpoint before it", index, segment.From)
		}
		if segment.Steps <= 0 || segment.Batch <= 0 || segment.Checkpoint == "" || segment.PeakDeviceBytes == 0 {
			t.Errorf("segment %d records no steps, batch, checkpoint or card memory", index)
		}
		if segment.HeldLossAfter < run.Segments[chosen].HeldLossAfter {
			chosen = index
		}
		seconds += segment.WallSeconds
	}
	// A segment whose held-back loss did not fall published nothing; its card-hours count all the same.
	for _, stopped := range run.Stopped {
		if stopped.From != run.Segments[len(run.Segments)-1].Checkpoint || stopped.Checkpoint != "" || stopped.HeldLossAfter < stopped.HeldLossBefore {
			t.Errorf("a stopped segment from %s published %q or lowered its held-back loss; want one past the last checkpoint that did neither", stopped.From, stopped.Checkpoint)
		}
		seconds += stopped.WallSeconds
	}
	if best := run.Segments[chosen]; run.Chosen != best.Checkpoint || !(best.HeldLossAfter < run.Segments[0].HeldLossBefore) {
		t.Errorf("the run chose %s; want %s, whose held-back loss %.4f is least, below its start's %.4f",
			run.Chosen, best.Checkpoint, best.HeldLossAfter, run.Segments[0].HeldLossBefore)
	}
	if spent, allowed := seconds/secondsPerHour, hours(terms); run.GPUHours < spent || run.GPUHours > allowed {
		t.Errorf("the run records %.2f card-hours over segments of %.2f; want them counted and within the terms' %.2f", run.GPUHours, spent, allowed)
	}
}
