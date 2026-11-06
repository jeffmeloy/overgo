package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// visionBridgeAcceptance is the vision-bridge experiment's terms, fixed before its first training step, and
// visionBridgeScoredPrefix the measure the scorecard reads a vision model's frame questions on, RxBrain's among them.
const (
	visionBridgeAcceptance   = "../../docs/verification/vision-bridge-acceptance.json"
	visionBridgeScoredPrefix = "vidgen-1m/frame-description-matching c8d91e306f4050ac2f8f319aeadd67efe3a8fce3e08ae07a81a03995d6962536"
)

// visionBridgeTerms is what the acceptance fixes that its check reads.
type visionBridgeTerms struct {
	Schema   string `json:"schema"`
	Question string `json:"question"`
	Suite    struct {
		Source       string `json:"source"`
		Kind         string `json:"kind"`
		Schema       string `json:"schema"`
		Seed         uint64 `json:"seed"`
		Cases        uint64 `json:"cases"`
		MaxTokens    uint64 `json:"max_tokens"`
		BuiltBy      string `json:"built_by"`
		Measure      string `json:"measure"`
		ScoredPrefix struct {
			Cases   uint64 `json:"cases"`
			Measure string `json:"measure"`
		} `json:"scored_prefix"`
		Names []string `json:"names"`
	} `json:"suite"`
	Split struct {
		HeldOut   string `json:"held_out"`
		Training  string `json:"training"`
		Selection string `json:"selection"`
	} `json:"split"`
	Comparisons []struct {
		Name      string `json:"name"`
		TextModel string `json:"text_model"`
		After     string `json:"after"`
	} `json:"comparisons"`
	Gain       float64 `json:"gain"`
	FalseGain  float64 `json:"false_gain"`
	MissedGain float64 `json:"missed_gain"`
	Share      float64 `json:"share"`
	Floor      string  `json:"floor"`
	Ceiling    struct {
		Model           string `json:"model"`
		Rule            string `json:"rule"`
		PeakDeviceBytes uint64 `json:"peak_device_bytes"`
	} `json:"ceiling"`
	Reference struct {
		Model             string  `json:"model"`
		Rule              string  `json:"rule"`
		ScoredPrefixShare float64 `json:"scored_prefix_share"`
	} `json:"reference"`
	Limits struct {
		TextModelPeakDeviceBytes    uint64  `json:"text_model_peak_device_bytes"`
		Composite9BPeakDeviceBytes  uint64  `json:"composite_9b_peak_device_bytes"`
		Composite05BPeakDeviceBytes uint64  `json:"composite_05b_peak_device_bytes"`
		AnswerSeconds               float64 `json:"answer_seconds"`
		GPUHours05B                 float64 `json:"gpu_hours_05b"`
		GPUHours9B                  float64 `json:"gpu_hours_9b"`
		GPUHoursReadings            float64 `json:"gpu_hours_readings"`
	} `json:"limits"`
	Order string   `json:"order"`
	Basis []string `json:"basis"`
}

// TestVisionBridgeAcceptanceIsFrozenBeforeTraining holds the vision-bridge experiment to the terms it fixed before
// training: a suite of distinct cases whose names give its measure and whose first cases are the frames the scorecard
// already reads, a required gain the suite can show and the scorecard's 64 cases cannot, a floor, a ceiling and a
// reference each named with how it is read, limits under the ceiling's card memory, the 0.5B composite first, and a
// training run record that names these terms by their digest, or none.
func TestVisionBridgeAcceptanceIsFrozenBeforeTraining(t *testing.T) {
	t.Parallel()
	data := testcheck.Must(os.ReadFile(visionBridgeAcceptance))(t)
	var terms visionBridgeTerms
	testcheck.NoError(t, strictjson.DecodeBytes(data, &terms))
	suite := terms.Suite
	if terms.Schema != "overgo/vision-bridge-acceptance/v1" || suite.Cases != uint64(len(suite.Names)) || suite.Seed == 0 || suite.BuiltBy == "" {
		t.Fatalf("the acceptance fixes no suite of its stated %d cases at a seed (%d names, schema %q)", suite.Cases, len(suite.Names), terms.Schema)
	}
	if sorted := slices.Sorted(slices.Values(suite.Names)); len(slices.Compact(sorted)) != len(suite.Names) {
		t.Error("the suite names a case twice")
	}
	if got := caseMeasure(suite.Source, suite.Names); got != suite.Measure {
		t.Errorf("the suite's names measure %s, the acceptance says %s", got, suite.Measure)
	}
	if prefix := suite.ScoredPrefix; prefix.Measure != visionBridgeScoredPrefix || caseMeasure(suite.Source, suite.Names[:prefix.Cases]) != prefix.Measure {
		t.Errorf("the suite's first %d cases are not the frames the scorecard reads (%s)", prefix.Cases, visionBridgeScoredPrefix)
	}
	comparisons := len(terms.Comparisons)
	detectable := func(cases uint64) float64 {
		return detectableGain(cases, terms.Share, terms.FalseGain/float64(comparisons), terms.MissedGain)
	}
	if comparisons == 0 || detectable(suite.Cases) > terms.Gain || detectable(suite.ScoredPrefix.Cases) <= terms.Gain {
		t.Errorf("a gain of %.3f is shown at %.3f on %d cases and %.3f on %d: the suite must show it and the scorecard's cases must not",
			terms.Gain, detectable(suite.Cases), suite.Cases, detectable(suite.ScoredPrefix.Cases), suite.ScoredPrefix.Cases)
	}
	if comparisons < 2 || !strings.Contains(terms.Comparisons[0].TextModel, "0.5B") || terms.Comparisons[1].After != terms.Comparisons[0].Name {
		t.Error("the 9B composite does not wait on the 0.5B composite's comparison")
	}
	limits := terms.Limits
	if !(limits.TextModelPeakDeviceBytes < limits.Composite9BPeakDeviceBytes && limits.Composite9BPeakDeviceBytes < terms.Ceiling.PeakDeviceBytes) ||
		limits.Composite05BPeakDeviceBytes == 0 || limits.AnswerSeconds <= 0 || limits.GPUHours05B <= 0 || limits.GPUHours9B <= 0 || limits.GPUHoursReadings <= 0 {
		t.Error("the limits do not hold the 9B composite between its text model's card memory and the ceiling's, or leave a limit unset")
	}
	if terms.Floor == "" || terms.Ceiling.Model == "" || terms.Ceiling.Rule == "" || terms.Reference.Rule == "" || len(terms.Basis) == 0 ||
		terms.Split.HeldOut == "" || terms.Split.Training == "" || terms.Split.Selection == "" {
		t.Error("the acceptance names no floor, ceiling, reference, split or basis")
	}
	digest := sha256.Sum256(data)
	runs := testcheck.Must(filepath.Glob(filepath.Join(filepath.Dir(visionBridgeAcceptance), "vision-bridge-*run*")))(t)
	for _, run := range runs {
		if !strings.Contains(string(testcheck.Must(os.ReadFile(run))(t)), hex.EncodeToString(digest[:])) {
			t.Errorf("%s trains under terms other than the acceptance's", filepath.Base(run))
		}
	}
}

// caseMeasure is the measure a suite of source asking cases of these names is read under, as evaluatedSet names it.
func caseMeasure(source string, names []string) string {
	asked := sha256.New()
	for _, name := range names {
		asked.Write([]byte(name + "\n"))
	}
	return source + " " + hex.EncodeToString(asked.Sum(nil))
}

// detectableGain is the least gain over a fixed floor a reading of cases shows with a one-sided false-gain rate and a
// missed-gain rate, for a share near share: the normal bound on a binomial share's error.
func detectableGain(cases uint64, share, falseGain, missedGain float64) float64 {
	quantile := func(tail float64) float64 { return math.Sqrt2 * math.Erfinv(1-2*tail) }
	return (quantile(falseGain) + quantile(missedGain)) * math.Sqrt(share*(1-share)/float64(cases))
}
