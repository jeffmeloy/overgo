package audioparity

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

func (f *audioMeasurementFixture) compareRepeats(t *testing.T, baseline audioMeasuredProcess) {
	t.Helper()
	repeat, candidate := f.measure(t), f.measure(t)
	for _, measured := range []audioMeasuredProcess{repeat, candidate} {
		if measured.Report.Plan != baseline.Report.Plan || measured.Report.Options != baseline.Report.Options || measured.Report.QualityScore != baseline.Report.QualityScore {
			t.Fatal("matched repeated execution changed protocol or held-out quality")
		}
		for index, execution := range measured.Report.Executions {
			before := baseline.Report.Executions[index]
			if execution.Name != before.Name || execution.Outcome != before.Outcome || execution.Failure != before.Failure || execution.Output != before.Output {
				t.Fatal("fresh repeated inference changed a source-bound output or failure")
			}
		}
	}
	lane := runrecord.ResourceFitnessLane{Name: "audio-evaluation-process", RequiredMetrics: []runrecord.ResourceMetric{runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes}, Baseline: baseline.Stream, Repeats: []runrecord.ObservationStream{repeat.Stream}, Candidate: candidate.Stream}
	comparison, err := runrecord.CompareResourceFitness(t.Context(), f.l.store, []runrecord.ResourceFitnessLane{lane})
	// A fresh unchanged-code run can regress beyond a small observed envelope.
	// Retain that refusal; neither cherry-pick a repeat nor widen the envelope.
	if err != nil {
		if !strings.Contains(err.Error(), "regresses metric") {
			t.Fatal(err)
		}
		content, contentErr := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-resource-refusal/v1"), struct {
			Baseline, Repeat, Candidate artifact.ID
			Reason                      string
		}{baseline.Report.ID, repeat.Report.ID, candidate.Report.ID, err.Error()})
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		batch, batchErr := artifact.NewDocumentBatch("audio-resource/refusal/"+content.Descriptor.ID.DigestHex(), []artifact.Content{content}, artifact.DependencyLineage(content.Descriptor.ID, baseline.Report.ID, repeat.Report.ID, candidate.Report.ID), nil)
		f.l.commit(t, batch, batchErr)
		t.Logf("fresh unchanged-code comparison refused: %v; no performance promotion", err)
	} else {
		batch, batchErr := comparison.VerifiedBatch(t.Context(), f.l.store, "audio-resource/comparison/"+comparison.ID.DigestHex())
		f.l.commit(t, batch, batchErr)
		replayed, replayErr := runrecord.RequireResourceFitnessComparison(t.Context(), f.l.store, comparison.ID)
		if replayErr != nil || replayed.ID != comparison.ID {
			t.Fatalf("fitness comparison replay: %v", replayErr)
		}
		t.Logf("resource comparison=%s observed_envelope=%+v strict_resource_separation=%t; same implementation and identical quality, no implementation improvement or promotion", comparison.ID, comparison.Lanes[0].NoiseEnvelope, comparison.StrictImprovement)
	}
	unknown := lane
	unknown.RequiredMetrics = append(slices.Clone(lane.RequiredMetrics), runrecord.ResourcePeakDeviceBytes)
	if _, err := runrecord.CompareResourceFitness(t.Context(), f.l.store, []runrecord.ResourceFitnessLane{unknown}); err == nil || !strings.Contains(err.Error(), "lacks required metric") {
		t.Fatalf("unmeasured device peak was admitted: %v", err)
	}
	reused := lane
	reused.Candidate = baseline.Stream
	if _, err := runrecord.CompareResourceFitness(t.Context(), f.l.store, []runrecord.ResourceFitnessLane{reused}); err == nil {
		t.Fatal("replayed baseline was admitted as a fresh candidate")
	}
	foreign := lane
	foreign.Candidate.Scope.Workload = baseline.Report.Dataset
	if _, err := runrecord.CompareResourceFitness(t.Context(), f.l.store, []runrecord.ResourceFitnessLane{foreign}); err == nil {
		t.Fatal("forged materialized scope was admitted")
	}
}

func TestAudioResourceEnvelopeAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": exact CPU resource batch acceptance executes measured child processes")
	}
	f := newAudioMeasurementFixture(t)
	f.compareRepeats(t, f.measure(t))
}

func TestAudioResourceFitnessAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": exact CPU fitness acceptance executes held-out, profile and repeated processes")
	}
	f := newAudioMeasurementFixture(t)
	baseline := f.measure(t)
	f.profile(t, baseline)
	f.compareRepeats(t, baseline)
	output := baselineCommand(t, f.l.fixture.root, "go", "test", "./internal/evaluation", "-run", "^TestAudioFitnessPolicyAcceptance$", "-count=1", "-v")
	if !strings.Contains(output, "--- PASS: TestAudioFitnessPolicyAcceptance") || strings.Contains(output, "--- SKIP:") {
		t.Fatalf("shared fitness policy was not executed:\n%s", output)
	}
	t.Log(output)
	t.Log("five held-out utterances, three speakers, known-duration silence and duration-unknown corrupt admission, real isolated-process peak/wall, separate profiled execution with exact output parity, fresh empirical envelopes and shared vector-policy refusals; no new runtime APIs, model state or buffers, no GPU/encoder-training/full-corpus/promotion claim")
}
