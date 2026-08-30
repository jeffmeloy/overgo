package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func prototypeProposal(t *testing.T) trainingprogram.ImprovementProposal {
	t.Helper()
	proposal, err := trainingprogram.CompileImprovementProposal(trainingprogram.ImprovementSpec{
		Kind:             trainingprogram.ImprovementModelPrototype,
		ParentModel:      testutil.ArtifactID(t, artifact.KindModel, "admission-parent"),
		Incumbent:        testutil.ArtifactID(t, artifact.KindProfile, "admission-incumbent"),
		Candidate:        testutil.ArtifactID(t, artifact.KindProfile, "admission-candidate"),
		Dataset:          testutil.ArtifactID(t, artifact.KindDataset, "admission-dataset"),
		DevelopmentSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "admission-development"),
		Recipe:           testutil.ArtifactID(t, artifact.KindRecipe, "admission-recipe"),
		Code:             testutil.ArtifactID(t, artifact.KindEvidence, "admission-code"),
		Proposer:         testutil.ArtifactID(t, artifact.KindEvidence, "admission-proposer"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return proposal
}

// TestModelPrototypeAdmissionContract pins the admission closure: a
// model-prototype proposal admits only with an incumbent to reject against,
// a promotion split independent of development, an objective recipe that can
// reject the hypothesis, and a positive resource ceiling — and the admitted
// record binds the motivating evidence, incumbent, and both splits exactly.
func TestModelPrototypeAdmissionContract(t *testing.T) {
	proposal := prototypeProposal(t)
	authority := testutil.ArtifactID(t, artifact.KindEvidence, "admission-authority")
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "admission-evaluator")
	promotion := testutil.ArtifactID(t, artifact.KindDatasetShard, "admission-promotion")
	objective := testutil.ArtifactID(t, artifact.KindRecipe, "admission-objective")

	admission, err := AdmitModelPrototype(proposal, authority, evaluator, promotion, objective, 30)
	if err != nil {
		t.Fatal(err)
	}
	if admission.Proposal != proposal.ID() || admission.Incumbent != proposal.Incumbent() ||
		admission.Candidate != proposal.Candidate() || admission.Code != proposal.Code() ||
		admission.DevelopmentSplit != proposal.DevelopmentSplit() || admission.PromotionSplit != promotion ||
		admission.Authority != authority {
		t.Fatalf("admission closure lost a binding: %+v", admission)
	}

	if _, err := AdmitModelPrototype(proposal, authority, evaluator, proposal.DevelopmentSplit(), objective, 30); err == nil ||
		!strings.Contains(err.Error(), "independent promotion split") {
		t.Fatalf("dependent promotion split admitted: %v", err)
	}
	if _, err := AdmitModelPrototype(proposal, authority, evaluator, promotion, objective, 0); err == nil ||
		!strings.Contains(err.Error(), "positive resource ceiling") {
		t.Fatalf("unbudgeted prototype admitted: %v", err)
	}
	unfalsifying := testutil.ArtifactID(t, artifact.KindEvidence, "objective-as-evidence")
	if _, err := AdmitModelPrototype(proposal, authority, evaluator, promotion, unfalsifying, 30); err == nil ||
		!strings.Contains(err.Error(), "objective that can reject") {
		t.Fatalf("non-rejecting objective admitted: %v", err)
	}
}
