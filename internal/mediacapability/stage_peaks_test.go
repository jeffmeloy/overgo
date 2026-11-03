package mediacapability

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/processmeasure"
	"overgo/internal/testutil"
)

// stagePeaksRecordPath is the retained record of the device memory each media family's requests held, by stage, and
// stagePeaksWriter the test that writes it.
const (
	stagePeaksRecordPath = "docs/verification/media-stage-peaks.json"
	stagePeaksWriter     = "OVERGO_CUDA_TEST=1 OVERGO_PROBE_TEST=1 go test -tags modeltest ./internal/mediacapability -run '^TestMediaStagePeaksAreRecorded$' -timeout 2h"
)

// stagePeaksRecord is what one run of the writer read: on which commit and card, and for each request a family is read
// at, its wall, the most device memory the process held during it, and each named part's.
type stagePeaksRecord struct {
	Commit   string              `json:"commit"`
	Device   string              `json:"device"`
	Protocol string              `json:"protocol"`
	Requests []stagePeaksRequest `json:"requests"`
}

type stagePeaksRequest struct {
	Family          string                `json:"family"`
	Label           string                `json:"label"`
	Request         json.RawMessage       `json:"request"`
	Wall            time.Duration         `json:"wall_ns"`
	PeakDeviceBytes uint64                `json:"peak_device_bytes"`
	Stages          []processmeasure.Part `json:"stages"`
}

// stagePeakRequests are the requests every media family is read at: Krea on a first picture, the same prompt on a new
// seed and a new prompt, on one session; MOSS at 1024 and 2048 square; Wan at the size a person asks for; LiveEdit on a
// clip of Wan's length at its source size; and the Wan autoencoder's round trip of the reconstruction evaluation's frame.
var stagePeakRequests = []struct{ family, label string }{
	{"krea", "first"}, {"krea", "repeat"}, {"krea", "new prompt"},
	{"moss", "1024"}, {"moss", "2048"},
	{"wan", "user size"}, {"liveedit", "81 frames"}, {"wan-vae", "round trip"},
}

// TestStagePeaksCoverEveryMediaFamily holds the retained record of the device memory by stage to every media family's
// requests: each was read on a commit the repository holds and a card it names, held device memory, and named parts of
// its path that held some, none above the request's own peak. A stage that changes is measured against
// these readings.
func TestStagePeaksCoverEveryMediaFamily(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var record stagePeaksRecord
	if err := jsonfile.DecodeStrict(filepath.Join(root, stagePeaksRecordPath), &record); err != nil {
		t.Fatalf("the record of stage peaks is unreadable (%v); write it with %s", err, stagePeaksWriter)
	}
	if _, err := gitauthority.Query(t.Context(), root, "cat-file", "-e", record.Commit+"^{commit}"); err != nil {
		t.Fatalf("commit %s: %v", record.Commit, err)
	}
	if record.Device == "" || record.Protocol == "" {
		t.Fatalf("the record does not state its card and protocol: %q, %q", record.Device, record.Protocol)
	}
	for _, want := range stagePeakRequests {
		at := slices.IndexFunc(record.Requests, func(read stagePeaksRequest) bool { return read.Family == want.family && read.Label == want.label })
		if at < 0 {
			t.Errorf("no reading of %s at %s", want.family, want.label)
			continue
		}
		read := record.Requests[at]
		if read.Wall <= 0 || read.PeakDeviceBytes == 0 || len(read.Request) == 0 {
			t.Errorf("%s at %s measured nothing: %+v", want.family, want.label, read)
		}
		if !slices.ContainsFunc(read.Stages, func(part processmeasure.Part) bool { return part.PeakDeviceBytes != 0 }) {
			t.Errorf("%s at %s names no part that held device memory: %+v", want.family, want.label, read.Stages)
		}
		for _, part := range read.Stages {
			if part.PeakDeviceBytes > read.PeakDeviceBytes || part.Wall < 0 || part.Times == 0 {
				t.Errorf("%s at %s: part %+v is no part of a request that held %d bytes at most", want.family, want.label, part, read.PeakDeviceBytes)
			}
		}
	}
}
