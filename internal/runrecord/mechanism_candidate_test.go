package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

type mechanismAdmissionFixture struct {
	store      *overgodb.Store
	candidate  trainingprogram.MechanismCandidate
	provenance externalMechanismProvenance
	binding    AdmissionBinding
	mechanism  artifact.ID
	census     trainingprogram.MechanismCensus
	cost       trainingprogram.MechanismEvidence
}

func TestMarinCandidateAdmissionRequiresRelevantTypedEvidence(t *testing.T) {
	fixture := newMechanismAdmissionFixture(t)
	defer fixture.store.Close()

	err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, fixture.candidate, fixture.provenance, fixture.binding.ID,
	)
	testcheck.NoError(t, err)

	foreign := fixture.provenance
	foreign.Census = testutil.ArtifactID(t, artifact.KindRecipe, "foreign-census")
	if err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, fixture.candidate, foreign, fixture.binding.ID,
	); err == nil {
		t.Fatal("foreign provenance admitted candidate")
	}
	if err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, fixture.candidate, fixture.provenance,
		testutil.ArtifactID(t, artifact.KindEvidence, "untyped-authority"),
	); err == nil {
		t.Fatal("untyped authority admitted candidate")
	}
}

func TestMechanismAdmissionRejectsUnrelatedEvidence(t *testing.T) {
	fixture := newMechanismAdmissionFixture(t)
	defer fixture.store.Close()

	foreignMechanism := testutil.ArtifactID(t, artifact.KindRecipe, "foreign-mechanism")
	foreignSource := testutil.ArtifactID(t, artifact.KindEvidence, "foreign-benefit-source")
	foreignBenefit := testcheck.Must(trainingprogram.NewMechanismEvidence(
		trainingprogram.MechanismEvidenceBenefit, foreignMechanism, foreignSource,
	))(t)
	foreignContent := testcheck.Must(foreignBenefit.Content())(t)
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key: "fixture/mechanism-admission/foreign-benefit",
		Artifacts: []artifact.Descriptor{
			{ID: foreignMechanism}, {ID: foreignSource},
		},
		Contents: []artifact.Content{foreignContent}, Lineage: foreignBenefit.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	unrelated := testcheck.Must(trainingprogram.CompileMechanismCandidate(
		fixture.census, fixture.provenance.ID, fixture.mechanism, fixture.cost.ID, foreignBenefit.ID,
	))(t)
	if err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, unrelated, fixture.provenance, fixture.binding.ID,
	); err == nil {
		t.Fatal("typed evidence for a foreign mechanism subject was admitted")
	}

	wrongRoleSource := testutil.ArtifactID(t, artifact.KindEvidence, "wrong-role-source")
	wrongRole := testcheck.Must(trainingprogram.NewMechanismEvidence(
		trainingprogram.MechanismEvidenceCostBound, fixture.mechanism, wrongRoleSource,
	))(t)
	wrongRoleContent := testcheck.Must(wrongRole.Content())(t)
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key:       "fixture/mechanism-admission/wrong-role",
		Artifacts: []artifact.Descriptor{{ID: wrongRoleSource}},
		Contents:  []artifact.Content{wrongRoleContent}, Lineage: wrongRole.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	wrongRoleCandidate := testcheck.Must(trainingprogram.CompileMechanismCandidate(
		fixture.census, fixture.provenance.ID, fixture.mechanism, fixture.cost.ID, wrongRole.ID,
	))(t)
	if err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, wrongRoleCandidate, fixture.provenance, fixture.binding.ID,
	); err == nil {
		t.Fatal("cost evidence used as benefit evidence was admitted")
	}

	descriptorOnly := testutil.ArtifactID(t, artifact.KindEvidence, "descriptor-only-benefit")
	if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
		Key:       "fixture/mechanism-admission/descriptor-only",
		Artifacts: []artifact.Descriptor{{ID: descriptorOnly}},
	}); err != nil {
		t.Fatal(err)
	}
	descriptorCandidate := testcheck.Must(trainingprogram.CompileMechanismCandidate(
		fixture.census, fixture.provenance.ID, fixture.mechanism, fixture.cost.ID, descriptorOnly,
	))(t)
	if err := ValidateMechanismCandidateEvidence(
		t.Context(), fixture.store, descriptorCandidate, fixture.provenance, fixture.binding.ID,
	); err == nil {
		t.Fatal("descriptor-only benefit evidence was admitted")
	}
}

func newMechanismAdmissionFixture(t *testing.T) mechanismAdmissionFixture {
	t.Helper()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	mechanism := id(artifact.KindRecipe, "quantile-router")
	claim := func(role trainingprogram.MechanismEvidenceRole, sourceKind artifact.Kind, name string) trainingprogram.MechanismEvidence {
		t.Helper()
		value := testcheck.Must(trainingprogram.NewMechanismEvidence(role, mechanism, id(sourceKind, name+"-source")))(t)
		return value
	}
	gap := claim(trainingprogram.MechanismEvidenceGap, artifact.KindEvidence, "gap")
	falsifier := claim(trainingprogram.MechanismEvidenceFalsifier, artifact.KindRecipe, "falsifier")
	cost := claim(trainingprogram.MechanismEvidenceCostBound, artifact.KindEvidence, "cost")
	benefit := claim(trainingprogram.MechanismEvidenceBenefit, artifact.KindEvidence, "benefit")
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: mechanism, Owner: id(artifact.KindEvidence, "router-owner"),
		Gap: gap.ID, Falsifier: falsifier.ID, GapState: trainingprogram.MechanismGapOpen,
	}
	census := testcheck.Must(trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment}))(t)
	provenance := testcheck.Must(NewExternalMechanismProvenance(
		census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", id(artifact.KindEvidence, "license"),
		[]ExternalMechanismSource{{Mechanism: mechanism, Paths: []string{"experiments/grug/moe/model.py"}}},
	))(t)
	candidate := testcheck.Must(trainingprogram.CompileMechanismCandidate(census, provenance.ID, mechanism, cost.ID, benefit.ID))(t)
	binding := testcheck.Must(NewAdmissionBinding(AdmissionBinding{
		Generation:   0,
		Proposer:     AuthorityDomain{Name: "mechanism-proposer", Identity: id(artifact.KindEvidence, "proposer")},
		Evaluator:    AuthorityDomain{Name: "mechanism-evaluator", Identity: id(artifact.KindEvidence, "evaluator")},
		Decider:      AuthorityDomain{Name: "mechanism-decider", Identity: id(artifact.KindEvidence, "decider")},
		SealedInputs: id(artifact.KindDataset, "sealed-inputs"),
		CleanWorker:  id(artifact.KindEvidence, "clean-worker"),
	}))(t)
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	claims := []trainingprogram.MechanismEvidence{gap, falsifier, cost, benefit}
	contents := make([]artifact.Content, 0, len(claims)+3)
	lineage := make([]artifact.Lineage, 0, len(claims)*2+len(census.Lineage())+len(provenance.Lineage()))
	descriptors := []artifact.Descriptor{
		{ID: mechanism}, {ID: assessment.Owner}, {ID: provenance.License},
		{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity}, {ID: binding.Decider.Identity},
		{ID: binding.SealedInputs}, {ID: binding.CleanWorker},
	}
	for _, evidence := range claims {
		content, contentErr := evidence.Content()
		if contentErr != nil {
			store.Close()
			t.Fatal(contentErr)
		}
		contents = append(contents, content)
		lineage = append(lineage, evidence.Lineage()...)
		descriptors = append(descriptors, artifact.Descriptor{ID: evidence.Source})
	}
	for _, contentValue := range []struct {
		content func() (artifact.Content, error)
		lineage []artifact.Lineage
	}{
		{census.Content, census.Lineage()},
		{provenance.Content, provenance.Lineage()},
		{binding.Content, nil},
	} {
		content, contentErr := contentValue.content()
		if contentErr != nil {
			store.Close()
			t.Fatal(contentErr)
		}
		contents = append(contents, content)
		lineage = append(lineage, contentValue.lineage...)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/mechanism-admission/authorities", Artifacts: descriptors, Contents: contents, Lineage: lineage,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	return mechanismAdmissionFixture{
		store: store, candidate: candidate, provenance: provenance, binding: binding,
		mechanism: mechanism, census: census, cost: cost,
	}
}
