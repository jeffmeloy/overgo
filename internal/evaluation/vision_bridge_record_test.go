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

// visionBridgeJudgement and visionBridgeNineBJudgement are the judgements of the 0.5B and the 9B composites.
const (
	visionBridgeJudgement      = "../../docs/verification/vision-bridge-0.5b-judgement.json"
	visionBridgeNineBJudgement = "../../docs/verification/vision-bridge-9b-judgement.json"
)

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

// TestVisionBridgeRecordIsComplete holds the 0.5B composite's judgement to the terms, its floor the composite whose
// bridge gives nothing, within the 0.5B composite's card memory.
func TestVisionBridgeRecordIsComplete(t *testing.T) {
	t.Parallel()
	judgementIsComplete(t, visionBridgeJudgement, visionBridgeRun, 0, []string{"floor"},
		func(terms visionBridgeTerms) uint64 { return terms.Limits.Composite05BPeakDeviceBytes })
}

// TestTheNineBJudgementIsComplete holds the 9B composite's judgement to the terms, its floor the higher of the
// composite whose bridge gives nothing and the run's start, the captions' mean, within the 9B composite's card memory.
func TestTheNineBJudgementIsComplete(t *testing.T) {
	t.Parallel()
	judgementIsComplete(t, visionBridgeNineBJudgement, visionBridgeNineBRun, 1, []string{"floor", "start"},
		func(terms visionBridgeTerms) uint64 { return terms.Limits.Composite9BPeakDeviceBytes })
}

// judgementIsComplete holds a composite's judgement at path to the terms it reads under: the frozen suite read whole by
// the trained composite, each untrained composite, the ceiling and the reference, each score its cases' share; the
// checkpoint the run at runPath chose; the floor the higher of chance and every untrained composite; the gain the
// composite's score less it; and a verdict of gain exactly when the gain reaches the terms' and the composite keeps
// within its card memory and the terms' answer time.
func judgementIsComplete(t *testing.T, path, runPath string, comparison int, untrained []string, memory func(visionBridgeTerms) uint64) {
	t.Helper()
	data := testcheck.Must(os.ReadFile(visionBridgeAcceptance))(t)
	var terms visionBridgeTerms
	testcheck.NoError(t, strictjson.DecodeBytes(data, &terms))
	digest := sha256.Sum256(data)
	var judgement visionBridgeJudgementRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(path))(t), &judgement))
	var run visionBridgeRunRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(runPath))(t), &run))
	if judgement.Schema != "overgo/vision-bridge-judgement/v1" || judgement.Acceptance != "sha256:"+hex.EncodeToString(digest[:]) ||
		judgement.Comparison != terms.Comparisons[comparison].Name || judgement.Measure != terms.Suite.Measure || judgement.Checkpoint != run.Chosen {
		t.Fatalf("the judgement reads under %s, comparison %q, measure %q and checkpoint %s; want the acceptance's, its comparison, its suite and the run's choice",
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
	floor := judgement.Chance
	for _, name := range append([]string{"composite", "ceiling", "reference"}, untrained...) {
		if _, read := readings[name]; !read {
			t.Errorf("the judgement holds no %s reading", name)
		}
	}
	for _, name := range untrained {
		floor = max(floor, readings[name].Score)
	}
	if judgement.Chance <= 0 || judgement.Floor != floor || judgement.Gain != readings["composite"].Score-floor {
		t.Errorf("the judgement sets the floor at %.4f and the gain at %.4f; want %.4f and %.4f", judgement.Floor, judgement.Gain, floor, readings["composite"].Score-floor)
	}
	gained := judgement.Gain >= terms.Gain && judgement.CompositePeakDeviceBytes > 0 && judgement.CompositePeakDeviceBytes <= memory(terms) &&
		judgement.CompositeAnswerSeconds > 0 && judgement.CompositeAnswerSeconds <= terms.Limits.AnswerSeconds
	if want := map[bool]string{true: "gain", false: "loss"}[gained]; judgement.Verdict != want {
		t.Errorf("the judgement's verdict is %q; want %q", judgement.Verdict, want)
	}
}
