package evaluation

import (
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/sequencescore"
	"overgo/internal/trainingprogram"
)

// TestUnpairedPreferenceObjectiveApprovesOnHeldOutGain judges a DPO
// objective that declares the unpaired metric on held-out completions,
// each labelled desirable or not. The plan binds every label and the KTO
// policy they derive; a trained policy whose rewards rise on desirable
// completions and fall on undesirable ones lowers the held-out KTO loss
// and passes, and its verdict commits and re-derives as passed from the
// scores its reports retain, while the opposite policy fails. Observations
// whose label or KL partner differs from the plan are refused.
func TestUnpairedPreferenceObjectiveApprovesOnHeldOutGain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	derived := func(kind artifact.Kind, role string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/unpaired-heldout-test/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	source := derived(artifact.KindDataset, "dataset")
	var records []dataset.Record
	for index := range 12 {
		records = append(records, dataset.Record{ID: fmt.Sprintf("completion/%02d", index), Group: fmt.Sprintf("prompt-%02d", index)})
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
	session := heldoutTraining(t, store, source, training.ID)
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "kto", Kind: trainingprogram.ObjectiveDPO,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricUnpairedPreferenceLoss, Evidence: []artifact.ID{session.Observation},
		Authority: trainingprogram.ObjectiveAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	view, err := CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]bool{}
	for index, record := range view.Records {
		labels[record.ID] = index%2 == 0
	}
	if len(view.Records) < 2 {
		t.Fatalf("the held-out view holds %d records, want desirable and undesirable ones", len(view.Records))
	}
	plan, err := CompileUnpairedHeldoutPlan(objective, view, derived(artifact.KindModel, "reference"), 0.1, labels)
	if err != nil {
		t.Fatal(err)
	}
	score := func(value float64) sequencescore.Score { return sequencescore.Score{LogProbability: value, Tokens: 1} }
	// observe answers every held-out record with its reward shift above
	// the reference when desirable and below it when not.
	observe := func(shift float64) []HeldoutObservation {
		var observations []HeldoutObservation
		for index, record := range view.Records {
			reward := shift
			if !labels[record.ID] {
				reward = -shift
			}
			observations = append(observations, HeldoutObservation{
				Record: record.ID, Partner: view.Records[(index+1)%len(view.Records)].ID, Unpaired: trainingprogram.UnpairedScores{
					Desirable: labels[record.ID], Policy: score(-2 + reward), Reference: score(-2),
					MismatchedPolicy: score(-4), MismatchedReference: score(-4),
				},
			})
		}
		return observations
	}
	reports := map[float64]HeldoutReport{}
	for _, shift := range []float64{0, 1, -1} {
		if reports[shift], err = plan.Score(observe(shift)); err != nil {
			t.Fatal(err)
		}
	}
	registration, err := split.PublicationBatch("test-registration", nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	registration.Contents = append(registration.Contents, content)
	for _, id := range []artifact.ID{source, objective.Processors[0], objective.Loss, objective.Evaluation, derived(artifact.KindModel, "reference")} {
		registration.Artifacts = append(registration.Artifacts, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, store, registration); err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []struct {
		name   string
		shift  float64
		passed bool
	}{{"preferring desirable completions", 1, true}, {"preferring undesirable completions", -1, false}} {
		verdict, err := JudgeHeldout(objective, view, plan, reports[0], reports[candidate.shift], session)
		if err != nil || verdict.Metric != trainingprogram.MetricUnpairedPreferenceLoss || verdict.Passed != candidate.passed {
			t.Fatalf("%s = (%+v, %v), want passed %t on held-out KTO loss", candidate.name, verdict, err, candidate.passed)
		}
		batch, err := HeldoutBatch(view, plan, reports[0], reports[candidate.shift], verdict)
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
		t.Logf("%s: held-out KTO loss %.6f -> %.6f over %d records", candidate.name, verdict.BaselineValue, verdict.CandidateValue, len(view.Records))
	}
	flipped, partnered := observe(1), observe(1)
	flipped[0].Unpaired.Desirable = !flipped[0].Unpaired.Desirable
	partnered[0].Partner = partnered[0].Record
	for name, observations := range map[string][]HeldoutObservation{"a flipped label": flipped, "a record its own partner": partnered} {
		if _, err := plan.Score(observations); err == nil {
			t.Fatalf("scored %s", name)
		}
	}
}
