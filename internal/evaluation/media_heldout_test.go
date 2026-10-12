package evaluation

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// TestMediaObjectiveApprovesOnHeldOutGain judges a tiny synthetic audio
// objective on held-out signal-to-noise: each held-out record expects a
// tone, the base model answers it under loud noise and the trained model
// under quiet noise. The objective declares a bounded held-out subset, so
// only its first held-out records are scored; the verdict passes on the SNR
// gain, commits, and re-derives as passed, which is what approves the
// objective, while a trained model no quieter than its base fails. An answer
// the ratio has no value for, because it equals its target or its target is
// silent, is refused with its record, the metric and the reason.
func TestMediaObjectiveApprovesOnHeldOutGain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	derived := func(kind artifact.Kind, role string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/media-heldout-test/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	source := derived(artifact.KindDataset, "dataset")
	var records []dataset.Record
	for index := range 12 {
		records = append(records, dataset.Record{ID: fmt.Sprintf("utterance/%02d", index), Group: fmt.Sprintf("speaker-%02d", index)})
	}
	split, err := dataset.BuildGroupSplit(source, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	var training, heldout dataset.Membership
	for _, membership := range split.Memberships {
		if membership.Partition == "train" {
			training = membership
		} else {
			heldout = membership
		}
	}
	const bound = 3
	session := heldoutTraining(t, store, source, training.ID)
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "speech", Kind: trainingprogram.ObjectiveLatentL2,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityAudio},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricAudioSNR, HeldoutRecords: bound, Evidence: []artifact.ID{session.Observation},
		Authority: trainingprogram.ObjectiveAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	view, err := CompileSFTEvaluationView(objective, training, heldout)
	if err != nil || len(heldout.Records) <= bound || len(view.Records) != bound || view.Records[0] != heldout.Records[0] {
		t.Fatalf("view holds %d of %d held-out records (%v), want the first %d", len(view.Records), len(heldout.Records), err, bound)
	}
	// Each record expects a tone of its own pitch; an answer is the tone
	// under a fixed pattern of noise at amplitude.
	const samples = 32
	tone := func(record int) []float64 {
		values := make([]float64, samples)
		for sample := range values {
			values[sample] = math.Sin(float64((record+1)*sample) / 4)
		}
		return values
	}
	var cases []HeldoutCase
	for index, record := range view.Records {
		cases = append(cases, HeldoutCase{Record: record.ID, Values: tone(index)})
	}
	plan, err := CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		t.Fatal(err)
	}
	answer := func(amplitude float64) HeldoutReport {
		t.Helper()
		var observations []HeldoutObservation
		for index, target := range cases {
			values := tone(index)
			for sample := range values {
				values[sample] += amplitude * math.Cos(float64(sample*sample))
			}
			observations = append(observations, HeldoutObservation{Record: target.Record, Values: values})
		}
		report, err := plan.Score(observations)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	base, trained, louder := answer(0.5), answer(0.1), answer(0.8)
	// The store holds the objective, its split and the identities it names,
	// as a registration commits them.
	registration, err := split.PublicationBatch("test-registration", nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	registration.Contents = append(registration.Contents, content)
	for _, id := range []artifact.ID{source, objective.Processors[0], objective.Loss, objective.Evaluation} {
		registration.Artifacts = append(registration.Artifacts, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, store, registration); err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []struct {
		name   string
		report HeldoutReport
		passed bool
	}{{"quieter", trained, true}, {"louder", louder, false}} {
		verdict, err := JudgeHeldout(objective, view, plan, base, candidate.report, session)
		if err != nil || verdict.Metric != trainingprogram.MetricAudioSNR || verdict.Passed != candidate.passed {
			t.Fatalf("%s trained answer = (%+v, %v), want passed %t on held-out SNR", candidate.name, verdict, err, candidate.passed)
		}
		batch, err := HeldoutBatch(view, plan, base, candidate.report, verdict)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		passed, err := CommittedHeldoutVerdictPassed(ctx, store, verdict.ID, objective)
		if err != nil || passed != candidate.passed {
			t.Fatalf("%s committed verdict = (%t, %v), want %t", candidate.name, passed, err, candidate.passed)
		}
		t.Logf("%s: held-out SNR %.2f dB -> %.2f dB over %d of %d held-out records", candidate.name, verdict.BaselineValue, verdict.CandidateValue, len(view.Records), len(heldout.Records))
	}
	// A produced signal the finite-values oracle fails is not measured.
	var broken []HeldoutObservation
	for _, target := range cases {
		broken = append(broken, HeldoutObservation{Record: target.Record, Values: slices.Clone(target.Values)})
	}
	broken[0].Values[0] = math.NaN()
	if _, err := plan.Score(broken); err == nil {
		t.Fatal("a produced signal holding a non-finite value was measured")
	}
	// A ratio in decibels has no value where the answer equals its target or
	// the target is silent. Every answer here equals the first plan's target,
	// and the second plan's first target is silent; answers are measured in
	// the order given, so each refusal is of the first record.
	silent := slices.Clone(cases)
	silent[0].Values = make([]float64, samples)
	quiet, err := CompileHeldoutPlan(objective, view, silent)
	if err != nil {
		t.Fatal(err)
	}
	var exact []HeldoutObservation
	for _, target := range cases {
		exact = append(exact, HeldoutObservation{Record: target.Record, Values: slices.Clone(target.Values)})
	}
	for _, refused := range []struct {
		plan   HeldoutPlan
		reason string
	}{{plan, "exactly"}, {quiet, "silent"}} {
		_, err := refused.plan.Score(exact)
		for _, part := range []string{cases[0].Record, string(trainingprogram.MetricAudioSNR), refused.reason} {
			if err == nil || !strings.Contains(err.Error(), part) {
				t.Errorf("a ratio with no value was refused as %q, want the refusal to hold %q", fmt.Sprint(err), part)
			}
		}
	}
}
