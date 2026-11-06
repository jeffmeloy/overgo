package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestRunMechanismPublishesValidatedEvidence(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	mechanism := id(artifact.KindRecipe, "mechanism")
	claim := func(role trainingprogram.MechanismEvidenceRole, sourceKind artifact.Kind, name string) trainingprogram.MechanismEvidence {
		t.Helper()
		value := testcheck.Must(trainingprogram.NewMechanismEvidence(role, mechanism, id(sourceKind, name+"-source")))(t)
		return value
	}
	gap := claim(trainingprogram.MechanismEvidenceGap, artifact.KindEvidence, "gap")
	falsifier := claim(trainingprogram.MechanismEvidenceFalsifier, artifact.KindRecipe, "falsifier")
	cost := claim(trainingprogram.MechanismEvidenceCostBound, artifact.KindEvidence, "cost-bound")
	benefit := claim(trainingprogram.MechanismEvidenceBenefit, artifact.KindEvidence, "benefit")
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: mechanism, Owner: id(artifact.KindEvidence, "owner"),
		Gap: gap.ID, Falsifier: falsifier.ID, GapState: trainingprogram.MechanismGapOpen,
	}
	census := testcheck.Must(trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment}))(t)
	license := id(artifact.KindEvidence, "license")
	provenance := testcheck.Must(runrecord.NewExternalMechanismProvenance(
		census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license,
		[]runrecord.ExternalMechanismSource{{Mechanism: assessment.Mechanism, Paths: []string{"experiments/grug/moe/model.py"}}},
	))(t)
	binding := testcheck.Must(runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation: 0,
		Proposer: runrecord.AuthorityDomain{
			Name: "mechanism-proposer", Identity: id(artifact.KindEvidence, "proposer"),
		},
		Evaluator: runrecord.AuthorityDomain{
			Name: "mechanism-evaluator", Identity: id(artifact.KindEvidence, "evaluator"),
		},
		Decider: runrecord.AuthorityDomain{
			Name: "mechanism-decider", Identity: id(artifact.KindEvidence, "decider"),
		},
		SealedInputs: id(artifact.KindDataset, "sealed-inputs"),
		CleanWorker:  id(artifact.KindEvidence, "clean-worker"),
	}))(t)
	specification := mechanismCandidateSpecification{
		Assessments: []trainingprogram.MechanismAssessment{assessment}, Provenance: provenance.ID,
		Mechanism: assessment.Mechanism, CostBound: cost.ID, Benefit: benefit.ID,
		Authority: binding.ID,
	}
	data := testcheck.Must(json.Marshal(specification))(t)
	root := t.TempDir()
	input := filepath.Join(root, "candidate.json")
	testcheck.NoError(t, os.WriteFile(input, data, 0o600))
	storePath := filepath.Join(root, "store")
	store := testcheck.Must(overgodb.Create(storePath))(t)
	descriptors := []artifact.Descriptor{
		{ID: assessment.Mechanism}, {ID: assessment.Owner}, {ID: license},
		{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity}, {ID: binding.Decider.Identity},
		{ID: binding.SealedInputs}, {ID: binding.CleanWorker},
	}
	claims := []trainingprogram.MechanismEvidence{gap, falsifier, cost, benefit}
	contents := make([]artifact.Content, 0, len(claims)+3)
	lineage := make([]artifact.Lineage, 0, len(claims)*2+len(census.Lineage())+len(provenance.Lineage()))
	for _, evidence := range claims {
		content, contentErr := evidence.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		contents = append(contents, content)
		lineage = append(lineage, evidence.Lineage()...)
		descriptors = append(descriptors, artifact.Descriptor{ID: evidence.Source})
	}
	censusContent := testcheck.Must(census.Content())(t)
	provenanceContent := testcheck.Must(provenance.Content())(t)
	bindingContent := testcheck.Must(binding.Content())(t)
	contents = append(contents, censusContent, provenanceContent, bindingContent)
	lineage = append(lineage, census.Lineage()...)
	lineage = append(lineage, provenance.Lineage()...)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/mechanism-admission/parents", Artifacts: descriptors,
		Contents: contents, Lineage: lineage,
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-mechanism", input, "-record", storePath}, &output))
	if !strings.Contains(output.String(), "mechanism candidate evidence validated:") {
		t.Fatalf("output = %q", output.String())
	}

	store = testcheck.Must(overgodb.Create(storePath))(t)
	foreignMechanism := id(artifact.KindRecipe, "foreign-mechanism")
	foreignSource := id(artifact.KindEvidence, "foreign-benefit-source")
	foreignBenefit := testcheck.Must(trainingprogram.NewMechanismEvidence(
		trainingprogram.MechanismEvidenceBenefit, foreignMechanism, foreignSource,
	))(t)
	foreignContent := testcheck.Must(foreignBenefit.Content())(t)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/mechanism-admission/foreign-benefit",
		Artifacts: []artifact.Descriptor{
			{ID: foreignMechanism}, {ID: foreignSource},
		},
		Contents: []artifact.Content{foreignContent}, Lineage: foreignBenefit.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	specification.Benefit = foreignBenefit.ID
	data = testcheck.Must(json.Marshal(specification))(t)
	testcheck.NoError(t, os.WriteFile(input, data, 0o600))
	if err := run([]string{"-mechanism", input, "-record", storePath}, &bytes.Buffer{}); err == nil {
		t.Fatal("command admitted benefit evidence for a foreign mechanism")
	}
}
