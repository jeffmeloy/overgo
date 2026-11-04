//go:build models

package evaluation

import (
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// mossDenoiserWallRecordPath holds where native MOSS's denoising wall goes, before and after the change it records.
const mossDenoiserWallRecordPath = "docs/verification/moss-denoiser-wall.json"

// mossDenoiserStepCounts are the two step counts whose integrate difference, over the steps between them, is the
// wall a step.
var mossDenoiserStepCounts = [2]int{4, 8}

// mossAccountedShare is how close a step's kernels and its wall must come for the wall to be accounted for: a captured
// step launches its kernels back to back, so what they leave is the device's own gaps, and the two are read in
// separate runs, either above the other by their spread.
const mossAccountedShare = 0.98

// mossDenoiserReading is one side of the change: its 50-step integrate and its integrate at the two step counts.
type mossDenoiserReading struct {
	IntegrateSeconds float64            `json:"integrate_seconds"`
	AtSteps          map[string]float64 `json:"integrate_seconds_at_steps"`
	Picture          artifact.ID        `json:"picture"`
	Kernels          map[string]float64 `json:"kernel_seconds_per_step"`
}

// perStep is the reading's wall a step: the integrate difference of the two step counts over the steps between them.
func (r mossDenoiserReading) perStep(t *testing.T) float64 {
	t.Helper()
	fewer, more := r.AtSteps[strconv.Itoa(mossDenoiserStepCounts[0])], r.AtSteps[strconv.Itoa(mossDenoiserStepCounts[1])]
	if fewer <= 0 || more <= fewer {
		t.Fatalf("the reading holds integrate walls %v at %v steps", r.AtSteps, mossDenoiserStepCounts)
	}
	return (more - fewer) / float64(mossDenoiserStepCounts[1]-mossDenoiserStepCounts[0])
}

// TestMOSSDenoiserWallIsAccountedFor holds the retained reading of native MOSS's denoising wall to its sources: the
// request is the resource protocol's first, its wall before the change the resource record's, its picture after the
// change the parity record's for that request bit for bit, a step faster after than before, and the step's kernels
// after the change holding its wall, so where the time beyond upstream's goes is read from the kernels named.
func TestMOSSDenoiserWallIsAccountedFor(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var record struct {
		BaseCommit string `json:"base_commit"`
		Device     string `json:"device"`
		Change     string `json:"change"`
		Request    struct {
			Label  string `json:"label"`
			Prompt string `json:"prompt"`
			Height int    `json:"height"`
			Width  int    `json:"width"`
			Seed   int64  `json:"seed"`
			Steps  int    `json:"steps"`
		} `json:"request"`
		Method   string              `json:"method"`
		Upstream float64             `json:"upstream_transformer_seconds"`
		Before   mossDenoiserReading `json:"before"`
		After    mossDenoiserReading `json:"after"`
		Scope    string              `json:"scope"`
	}
	if err := jsonfile.DecodeStrict(filepath.Join(root, mossDenoiserWallRecordPath), &record); err != nil {
		t.Fatal(err)
	}
	if _, err := gitauthority.Query(t.Context(), root, "cat-file", "-e", record.BaseCommit+"^{commit}"); err != nil {
		t.Fatalf("commit %s: %v", record.BaseCommit, err)
	}
	var resources mossResourceRecord
	if err := jsonfile.DecodeStrict(filepath.Join(root, mossResourceRecordPath), &resources); err != nil {
		t.Fatal(err)
	}
	native, upstream := resources.Native.Requests[0], resources.Upstream.Requests[0]
	request := record.Request
	if request.Label != native.Label || request.Prompt != native.Prompt || request.Height != native.Height ||
		request.Width != native.Width || request.Seed != native.Seed || record.Device != resources.Device {
		t.Fatalf("the reading's request %+v on %s is not the resource record's first, %+v on %s", request, record.Device, native, resources.Device)
	}
	if record.Before.IntegrateSeconds != native.Phases["integrate"] || record.Upstream != upstream.Phases["transformer"] {
		t.Fatalf("the reading's walls before, native %g and upstream %g, are not the resource record's %g and %g",
			record.Before.IntegrateSeconds, record.Upstream, native.Phases["integrate"], upstream.Phases["transformer"])
	}
	var parity mossParityRecord
	if err := jsonfile.DecodeStrict(filepath.Join(root, mossParityRecordPath), &parity); err != nil {
		t.Fatal(err)
	}
	pictured := false
	for _, compared := range parity.Cases {
		if compared.Prompt == request.Prompt && compared.Negative == "" && compared.Height == request.Height &&
			compared.Width == request.Width && compared.Seed == request.Seed {
			pictured = compared.Image == record.After.Picture
		}
	}
	if !pictured {
		t.Fatalf("the picture after the change, %s, is not the parity record's for the request", record.After.Picture)
	}
	before, after := record.Before.perStep(t), record.After.perStep(t)
	if after >= before || record.After.IntegrateSeconds >= record.Before.IntegrateSeconds {
		t.Fatalf("a step took %.3f s after the change against %.3f s before, the picture's integrate %.1f s against %.1f s",
			after, before, record.After.IntegrateSeconds, record.Before.IntegrateSeconds)
	}
	held := 0.0
	for _, seconds := range record.After.Kernels {
		held += seconds
	}
	if held > after/mossAccountedShare || held < mossAccountedShare*after {
		t.Fatalf("a step's kernels hold %.4f s of its %.4f s wall", held, after)
	}
	t.Logf("a step: %.3f s before, %.3f s after, upstream %.3f s; its kernels hold %.3f of it",
		before, after, record.Upstream/float64(request.Steps), held/after)
}
