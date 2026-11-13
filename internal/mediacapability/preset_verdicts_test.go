package mediacapability

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// presetVerdictsRecord holds the verdicts of the guidance presets' readings: each preset's judged suite against its
// model's reference at a person's size, paired question by question.
var presetVerdictsRecord = filepath.Join("..", "..", "docs", "verification", "guidance-preset-readings.json")

// TestTheReadPresetsHoldTheirVerdicts holds the presets a page offers to the readings' verdicts
// (guidance-ops/image-readings): every preset the readings name as admitted or refused has its verdict in the record,
// its status the verdict's, admitted only on no loss beyond chance, and every verdict's pairs are its wins, losses and
// ties, read against a reference and an arm the record names.
func TestTheReadPresetsHoldTheirVerdicts(t *testing.T) {
	t.Parallel()
	var readings map[string]map[string]string
	testcheck.NoError(t, json.Unmarshal(presetReadingsJSON, &readings))
	var record struct {
		Readings []struct {
			Family, Preset, Reference, Arm string
			Pairs, Wins, Losses, Ties      int
			Tail                           float64
			Admitted                       bool
			Says                           string
		}
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(presetVerdictsRecord))(t), &record))
	verdicts := map[string]bool{}
	for _, read := range record.Readings {
		key := read.Family + "/" + read.Preset
		if read.Wins+read.Losses+read.Ties != read.Pairs || read.Pairs == 0 || read.Reference == "" || read.Arm == "" || read.Says == "" {
			t.Errorf("the verdict of %s holds %d pairs of %d wins, %d losses and %d ties, against reference %q and arm %q, saying %q", key, read.Pairs, read.Wins, read.Losses, read.Ties, read.Reference, read.Arm, read.Says)
		}
		want := presetRefused
		if read.Admitted {
			want = presetAdmitted
		}
		if status := readings[read.Family][read.Preset]; status != want {
			t.Errorf("the readings hold %s as %q; its verdict says %s", key, status, want)
		}
		verdicts[key] = true
	}
	for family, presets := range readings {
		for preset, status := range presets {
			if status != presetPending && !verdicts[family+"/"+preset] {
				t.Errorf("the readings hold %s/%s as %s with no verdict in the record", family, preset, status)
			}
		}
	}
}
