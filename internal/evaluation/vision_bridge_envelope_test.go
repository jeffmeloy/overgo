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

// visionBridgeNineBEnvelope is the 9B composite's promotion envelope.
const visionBridgeNineBEnvelope = "../../docs/verification/vision-bridge-9b-envelope.json"

// visionBridgeEnvelopeReading is one reading an envelope's trial holds: its evaluation in the store, the cases it
// answered rightly of those it was asked, and its score.
type visionBridgeEnvelopeReading struct {
	Evaluation string  `json:"evaluation"`
	Correct    int     `json:"correct"`
	Cases      int     `json:"cases"`
	Score      float64 `json:"score"`
}

// visionBridgeEnvelopeTrial is one independent seed's trial: the seed its bridge's run ordered its samples by, the
// checkpoint that run chose and the adapter it publishes, its readings of the frame suite, the bridge's, the cheap
// baseline's, the dropped source's and the shuffled source's, the text regression pair, and the latency and card
// memory of the text model alone and composed.
type visionBridgeEnvelopeTrial struct {
	Seed                    uint64                      `json:"seed"`
	Checkpoint              string                      `json:"checkpoint"`
	Adapter                 string                      `json:"adapter"`
	Bridge                  visionBridgeEnvelopeReading `json:"bridge"`
	CheapBaseline           visionBridgeEnvelopeReading `json:"cheap_baseline"`
	DroppedSource           visionBridgeEnvelopeReading `json:"dropped_source"`
	ShuffledSource          visionBridgeEnvelopeReading `json:"shuffled_source"`
	RegressionBaseline      visionBridgeEnvelopeReading `json:"regression_baseline"`
	RegressionCandidate     visionBridgeEnvelopeReading `json:"regression_candidate"`
	BaselineLatencyNS       uint64                      `json:"baseline_latency_ns"`
	ComposedLatencyNS       uint64                      `json:"composed_latency_ns"`
	BaselinePeakDeviceBytes uint64                      `json:"baseline_peak_device_bytes"`
	ComposedPeakDeviceBytes uint64                      `json:"composed_peak_device_bytes"`
}

// visionBridgeEnvelopeRecord is the envelope: the terms it reads under by their digest, the comparison it serves, the
// frame suite's measure and file and its shuffled control's file, the text regression set and its cases, and each seed's
// trial.
type visionBridgeEnvelopeRecord struct {
	Schema        string                      `json:"schema"`
	Acceptance    string                      `json:"acceptance"`
	Comparison    string                      `json:"comparison"`
	Measure       string                      `json:"measure"`
	Suite         string                      `json:"suite"`
	ShuffledSuite string                      `json:"shuffled_suite"`
	Regression    visionBridgeRegressionSet   `json:"regression"`
	Trials        []visionBridgeEnvelopeTrial `json:"trials"`
	Notes         []string                    `json:"notes"`
}

// visionBridgeRegressionSet is the text set the regression pair reads: the derived suite and its cases.
type visionBridgeRegressionSet struct {
	Suite string `json:"suite"`
	Cases int    `json:"cases"`
}

// TestTheNineBEnvelopeIsComplete holds the 9B composite's envelope to the promotion's trial contract and the terms:
// two independent seeds or more, each its own; the first the judged run's, its checkpoint the run's choice and its
// bridge and dropped source the judgement's readings; every reading of the frame suite, its shuffled control or the
// regression set whole, each score its cases' share; and every resource measured.
func TestTheNineBEnvelopeIsComplete(t *testing.T) {
	t.Parallel()
	const minimumSeeds = 2
	data := testcheck.Must(os.ReadFile(visionBridgeAcceptance))(t)
	var terms visionBridgeTerms
	testcheck.NoError(t, strictjson.DecodeBytes(data, &terms))
	digest := sha256.Sum256(data)
	var envelope visionBridgeEnvelopeRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(visionBridgeNineBEnvelope))(t), &envelope))
	var run visionBridgeRunRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(visionBridgeNineBRun))(t), &run))
	var judgement visionBridgeJudgementRecord
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(visionBridgeNineBJudgement))(t), &judgement))
	if envelope.Schema != "overgo/vision-bridge-envelope/v1" || envelope.Acceptance != "sha256:"+hex.EncodeToString(digest[:]) ||
		envelope.Comparison != terms.Comparisons[1].Name || envelope.Measure != terms.Suite.Measure ||
		envelope.Suite == "" || envelope.ShuffledSuite == "" || envelope.ShuffledSuite == envelope.Suite || envelope.Regression.Suite == "" || envelope.Regression.Cases <= 0 {
		t.Fatalf("the envelope reads under %s, comparison %q, measure %q, suites %q and %q and regression %+v; want the acceptance's, the 9B's comparison, its suite, a shuffled control and a regression set",
			envelope.Acceptance, envelope.Comparison, envelope.Measure, envelope.Suite, envelope.ShuffledSuite, envelope.Regression)
	}
	if len(envelope.Trials) < minimumSeeds {
		t.Fatalf("the envelope holds %d trials; want a seed's for each of %d independent seeds", len(envelope.Trials), minimumSeeds)
	}
	readings := map[string]visionBridgeReading{}
	for _, reading := range judgement.Readings {
		readings[reading.Name] = reading
	}
	first := envelope.Trials[0]
	if first.Seed != 0 || first.Checkpoint != run.Chosen || first.Bridge.Evaluation != readings["composite"].Evaluation || first.DroppedSource.Evaluation != readings["floor"].Evaluation {
		t.Errorf("the first trial is seed %d's from %s; want the judged run's, its chosen checkpoint and the judgement's composite and floor readings", first.Seed, first.Checkpoint)
	}
	seeds := map[uint64]bool{}
	for index, trial := range envelope.Trials {
		if seeds[trial.Seed] || trial.Checkpoint == "" || trial.Adapter == "" {
			t.Errorf("trial %d is seed %d from %q, adapter %q; want a seed of its own, its run's checkpoint and its adapter", index, trial.Seed, trial.Checkpoint, trial.Adapter)
		}
		seeds[trial.Seed] = true
		for name, reading := range map[string]struct {
			read  visionBridgeEnvelopeReading
			cases int
		}{
			"bridge": {trial.Bridge, int(terms.Suite.Cases)}, "cheap baseline": {trial.CheapBaseline, int(terms.Suite.Cases)},
			"dropped source": {trial.DroppedSource, int(terms.Suite.Cases)}, "shuffled source": {trial.ShuffledSource, int(terms.Suite.Cases)},
			"regression baseline": {trial.RegressionBaseline, envelope.Regression.Cases}, "regression candidate": {trial.RegressionCandidate, envelope.Regression.Cases},
		} {
			const tolerance = 1e-9
			if read := reading.read; read.Evaluation == "" || read.Cases != reading.cases || read.Correct < 0 || read.Correct > read.Cases ||
				math.Abs(read.Score-float64(read.Correct)/float64(read.Cases)) > tolerance {
				t.Errorf("trial %d's %s reading answers %d of %d for %.4f; want %d cases, its score their share and its evaluation", index, name, read.Correct, read.Cases, read.Score, reading.cases)
			}
		}
		if trial.BaselineLatencyNS == 0 || trial.ComposedLatencyNS == 0 || trial.BaselinePeakDeviceBytes == 0 || trial.ComposedPeakDeviceBytes == 0 {
			t.Errorf("trial %d measures latencies %d and %d ns and card memory %d and %d bytes; want each measured", index,
				trial.BaselineLatencyNS, trial.ComposedLatencyNS, trial.BaselinePeakDeviceBytes, trial.ComposedPeakDeviceBytes)
		}
	}
}
