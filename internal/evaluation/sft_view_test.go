package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestSFTEvaluationViewIsHeldOutAndAuthorityBound(t *testing.T) {
	t.Parallel()
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "SFT view dataset")
	training := testcheck.Must(fixtureMembership(datasetID, "fit", dataset.Record{ID: "training", Group: "training-group"}))(t)
	heldout := testcheck.Must(fixtureMembership(datasetID, "heldout", dataset.Record{ID: "heldout", Group: "heldout-group"}))(t)
	processor := testutil.ArtifactID(t, artifact.KindProfile, "SFT view processor")
	projector := testutil.ArtifactID(t, artifact.KindProjector, "SFT view projector")
	codec := testutil.ArtifactID(t, artifact.KindProfile, "SFT view codec")
	objective := testcheck.Must(trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "held-out fixture", Kind: trainingprogram.ObjectiveOCR,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityImage},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: datasetID, Split: training.ID, Processors: []artifact.ID{processor},
		Projectors: []artifact.ID{projector}, Codecs: []artifact.ID{codec},
		Loss:       testutil.ArtifactID(t, artifact.KindProfile, "SFT view loss"),
		Evaluation: testutil.ArtifactID(t, artifact.KindProfile, "SFT view evaluation"),
		Metric:     trainingprogram.MetricTokenAccuracy,
		Evidence:   []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "SFT view evidence")},
		Authority:  trainingprogram.ObjectiveApproved,
	}))(t)
	view := testcheck.Must(CompileSFTEvaluationView(objective, training, heldout))(t)
	if view.Objective != objective.ID || view.TrainingMembership != training.ID || view.HeldoutMembership != heldout.ID ||
		view.Signature.Inputs[0] != recipecontract.ModalityImage || view.Processors[0] != processor ||
		view.Projectors[0] != projector || view.Codecs[0] != codec {
		t.Fatalf("view authority = %+v", view)
	}
	overlap := testcheck.Must(fixtureMembership(datasetID, "overlap", dataset.Record{ID: "other", Group: training.Records[0].Group}))(t)
	if _, err := CompileSFTEvaluationView(objective, training, overlap); err == nil {
		t.Fatal("group-overlapping held-out membership accepted")
	}
}

func fixtureMembership(source artifact.ID, name string, record dataset.Record) (dataset.Membership, error) {
	plan, err := dataset.BuildGroupSplit(source, []dataset.Record{record}, 11, []dataset.SplitPartition{
		{Name: name, Weight: 1}, {Name: name + "-unused", Weight: 1},
	})
	if err != nil {
		return dataset.Membership{}, err
	}
	for _, membership := range plan.Memberships {
		if len(membership.Records) != 0 {
			return membership, nil
		}
	}
	return dataset.Membership{}, nil
}
