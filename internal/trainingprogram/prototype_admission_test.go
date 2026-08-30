package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestModelPrototypeAdmissionContract pins the proposal side of the closure:
// a model-prototype proposal's candidate is the declared prototype profile
// document, the incumbent shares that kind and differs from the candidate,
// and a proposal whose candidate is not a profile identity refuses — so no
// prototype enters admission with the wrong artifact class.
func TestModelPrototypeAdmissionContract(t *testing.T) {
	spec := ImprovementSpec{
		Kind:             ImprovementModelPrototype,
		ParentModel:      testutil.ArtifactID(t, artifact.KindModel, "proposal-parent"),
		Incumbent:        testutil.ArtifactID(t, artifact.KindProfile, "proposal-incumbent"),
		Candidate:        testutil.ArtifactID(t, artifact.KindProfile, "proposal-candidate"),
		Dataset:          testutil.ArtifactID(t, artifact.KindDataset, "proposal-dataset"),
		DevelopmentSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "proposal-development"),
		Recipe:           testutil.ArtifactID(t, artifact.KindRecipe, "proposal-recipe"),
		Code:             testutil.ArtifactID(t, artifact.KindEvidence, "proposal-code"),
		Proposer:         testutil.ArtifactID(t, artifact.KindEvidence, "proposal-proposer"),
	}
	proposal, err := CompileImprovementProposal(spec)
	if err != nil {
		t.Fatal(err)
	}
	if proposal.ProposalKind() != ImprovementModelPrototype || proposal.Candidate() != spec.Candidate {
		t.Fatalf("prototype proposal = %+v", proposal)
	}

	wrongClass := spec
	wrongClass.Candidate = testutil.ArtifactID(t, artifact.KindModel, "prototype-as-model")
	wrongClass.Incumbent = testutil.ArtifactID(t, artifact.KindModel, "incumbent-as-model")
	if _, err := CompileImprovementProposal(wrongClass); err == nil {
		t.Fatal("prototype proposal with a non-profile candidate compiled")
	}
	selfIncumbent := spec
	selfIncumbent.Incumbent = spec.Candidate
	if _, err := CompileImprovementProposal(selfIncumbent); err == nil {
		t.Fatal("prototype proposal judging itself as incumbent compiled")
	}
}
