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
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestRunMechanismPublishesEvidenceGatedAdmission(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	mechanism := id(artifact.KindRecipe, "mechanism")
	claim := func(role trainingprogram.MechanismEvidenceRole, sourceKind artifact.Kind, name string) trainingprogram.MechanismEvidence {
		t.Helper()
		value, err := trainingprogram.NewMechanismEvidence(role, mechanism, id(sourceKind, name+"-source"))
		if err != nil {
			t.Fatal(err)
		}
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
	census, err := trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment})
	if err != nil {
		t.Fatal(err)
	}
	license := id(artifact.KindEvidence, "license")
	provenance, err := runrecord.NewExternalMechanismProvenance(
		census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license,
		[]runrecord.ExternalMechanismSource{{Mechanism: assessment.Mechanism, Paths: []string{"experiments/grug/moe/model.py"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
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
	})
	if err != nil {
		t.Fatal(err)
	}
	specification := mechanismCandidateSpecification{
		Assessments: []trainingprogram.MechanismAssessment{assessment}, Provenance: provenance.ID,
		Mechanism: assessment.Mechanism, CostBound: cost.ID, Benefit: benefit.ID,
		Authority: binding.ID,
	}
	data, err := json.Marshal(specification)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	input := filepath.Join(root, "candidate.json")
	if err := os.WriteFile(input, data, 0o600); err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
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
	censusContent, err := census.Content()
	if err != nil {
		t.Fatal(err)
	}
	provenanceContent, err := provenance.Content()
	if err != nil {
		t.Fatal(err)
	}
	bindingContent, err := binding.Content()
	if err != nil {
		t.Fatal(err)
	}
	contents = append(contents, censusContent, provenanceContent, bindingContent)
	lineage = append(lineage, census.Lineage()...)
	lineage = append(lineage, provenance.Lineage()...)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/mechanism-admission/parents", Artifacts: descriptors,
		Contents: contents, Lineage: lineage,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-mechanism", input, "-record", storePath}, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "mechanism candidate admitted:") {
		t.Fatalf("output = %q", output.String())
	}

	store, err = overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	foreignMechanism := id(artifact.KindRecipe, "foreign-mechanism")
	foreignSource := id(artifact.KindEvidence, "foreign-benefit-source")
	foreignBenefit, err := trainingprogram.NewMechanismEvidence(
		trainingprogram.MechanismEvidenceBenefit, foreignMechanism, foreignSource,
	)
	if err != nil {
		t.Fatal(err)
	}
	foreignContent, err := foreignBenefit.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/mechanism-admission/foreign-benefit",
		Artifacts: []artifact.Descriptor{
			{ID: foreignMechanism}, {ID: foreignSource},
		},
		Contents: []artifact.Content{foreignContent}, Lineage: foreignBenefit.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	specification.Benefit = foreignBenefit.ID
	data, err = json.Marshal(specification)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(input, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-mechanism", input, "-record", storePath}, &bytes.Buffer{}); err == nil {
		t.Fatal("command admitted benefit evidence for a foreign mechanism")
	}
}
