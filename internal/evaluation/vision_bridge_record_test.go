package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// visionBridgeJudgement is the judgement of the 0.5B composite, and visionBridgeReadings the readings it sets side by
// side: the trained composite, its floor, the ceiling and the reference.
const visionBridgeJudgement = "../../docs/verification/vision-bridge-0.5b-judgement.json"

var visionBridgeReadings = []string{"composite", "floor", "ceiling", "reference"}

// visionBridgeReading is one reading of the frozen suite: what was read, its evaluation in the store, the cases it
// answered rightly of those it was asked, its score, and the seconds a case took.
type visionBridgeReading struct {
	Name           string  `json:"name"`
	Model          string  `json:"model"`
	Projection     string  `json:"projection"`
	Evaluation     string  `json:"evaluation"`
	Correct        int     `json:"correct"`
	Cases          int     `json:"cases"`
	Score          float64 `json:"score"`
	SecondsPerCase float64 `json:"seconds_per_case"`
}

// visionBridgeJudgementRecord is the judgement: the terms it reads under by their digest, the comparison, the suite, the
// checkpoint the run chose, each reading, chance and the floor it sets, the gain over the floor, the composite's card
// memory and answer time against the terms' limits, and the verdict.
type visionBridgeJudgementRecord struct {
	Schema                   string                `json:"schema"`
	Acceptance               string                `json:"acceptance"`
	Comparison               string                `json:"comparison"`
	Measure                  string                `json:"measure"`
	Checkpoint               string                `json:"checkpoint"`
	Readings                 []visionBridgeReading `json:"readings"`
	Chance                   float64               `json:"chance"`
	Floor                    float64               `json:"floor"`
	Gain                     float64               `json:"gain"`
	CompositePeakDeviceBytes uint64                `json:"composite_peak_device_bytes"`
	CompositeAnswerSeconds   float64               `json:"composite_answer_seconds"`
	Verdict                  string                `json:"verdict"`
	Notes                    []string              `json:"notes"`
}

// TestVisionBridgeRecordIsComplete holds the 0.5B composite's judgement to the terms it reads under: the frozen suite
// read whole by the trained composite, its floor, the ceiling and the reference, each score its cases' share; the
// checkpoint the run chose; the floor the higher of chance and the untrained composite; the gain the composite's score
// less it; and a verdict of gain exactly when the gain reaches the terms' and the composite keeps within their card
// memory and answer time.
func TestVisionBridgeRecordIsComplete(t *testing.T) {
	t.Parallel()
	data := testcheck.Must(os.ReadFile(visionBridgeAcceptance))(t)
	var terms visionBridgeTerms
	testcheck.NoError(t, strictjson.DecodeBytes(data, &terms))
	digest := sha256.Sum256(data)
	var judgement visionBridgeJudgementRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(visionBridgeJudgement))(t), &judgement))
	var run visionBridgeRunRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(visionBridgeRun))(t), &run))
	if judgement.Schema != "overgo/vision-bridge-judgement/v1" || judgement.Acceptance != "sha256:"+hex.EncodeToString(digest[:]) ||
		judgement.Comparison != terms.Comparisons[0].Name || judgement.Measure != terms.Suite.Measure || judgement.Checkpoint != run.Chosen {
		t.Fatalf("the judgement reads under %s, comparison %q, measure %q and checkpoint %s; want the acceptance's, its first comparison, its suite and the run's choice",
			judgement.Acceptance, judgement.Comparison, judgement.Measure, judgement.Checkpoint)
	}
	readings := map[string]visionBridgeReading{}
	for _, reading := range judgement.Readings {
		const tolerance = 1e-9
		if reading.Cases != int(terms.Suite.Cases) || reading.Correct < 0 || reading.Correct > reading.Cases ||
			math.Abs(reading.Score-float64(reading.Correct)/float64(reading.Cases)) > tolerance || reading.Evaluation == "" || reading.Model == "" || reading.SecondsPerCase <= 0 {
			t.Errorf("the %s reading answers %d of %d cases for %.4f; want the suite's %d cases, its score their share, its evaluation, model and time",
				reading.Name, reading.Correct, reading.Cases, reading.Score, terms.Suite.Cases)
		}
		readings[reading.Name] = reading
	}
	for _, name := range visionBridgeReadings {
		if _, read := readings[name]; !read {
			t.Errorf("the judgement holds no %s reading", name)
		}
	}
	floor := max(judgement.Chance, readings["floor"].Score)
	if judgement.Chance <= 0 || judgement.Floor != floor || judgement.Gain != readings["composite"].Score-floor {
		t.Errorf("the judgement sets the floor at %.4f and the gain at %.4f; want %.4f and %.4f", judgement.Floor, judgement.Gain, floor, readings["composite"].Score-floor)
	}
	gained := judgement.Gain >= terms.Gain && judgement.CompositePeakDeviceBytes > 0 && judgement.CompositePeakDeviceBytes <= terms.Limits.Composite05BPeakDeviceBytes &&
		judgement.CompositeAnswerSeconds > 0 && judgement.CompositeAnswerSeconds <= terms.Limits.AnswerSeconds
	if want := map[bool]string{true: "gain", false: "loss"}[gained]; judgement.Verdict != want {
		t.Errorf("the judgement's verdict is %q; want %q", judgement.Verdict, want)
	}
}
