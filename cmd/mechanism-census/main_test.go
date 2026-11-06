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
	"overgo/internal/overgodbtest"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestRunPublishesMechanismCensus(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "mechanism"), Owner: id(artifact.KindEvidence, "owner"),
		Gap: id(artifact.KindEvidence, "gap"), Falsifier: id(artifact.KindRecipe, "falsifier"),
		GapState: trainingprogram.MechanismGapOpen,
	}
	specification, err := json.Marshal(censusSpecification{Assessments: []trainingprogram.MechanismAssessment{assessment}})
	testcheck.NoError(t, err)
	root := t.TempDir()
	input := filepath.Join(root, "census.json")
	testcheck.NoError(t, os.WriteFile(input, specification, 0o600))
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Create(storePath)
	testcheck.NoError(t, err)
	parents := []artifact.Descriptor{{ID: assessment.Mechanism}, {ID: assessment.Owner}, {ID: assessment.Gap}, {ID: assessment.Falsifier}}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/mechanism-census/parents", Artifacts: parents}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-input", input, "-store", storePath}, &output))
	if !strings.Contains(output.String(), "assessments=1 open=1 closed=0") {
		t.Fatalf("output = %q", output.String())
	}
	census, err := trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment})
	testcheck.NoError(t, err)
	store = overgodbtest.OpenReadOnly(t, storePath)
	if _, found, err := artifact.ReadContent(t.Context(), store, census.ID()); err != nil || !found {
		t.Fatalf("published census = (%t, %v)", found, err)
	}
}

func TestRunPublishesExternalMechanismProvenance(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "mechanism"), Owner: id(artifact.KindEvidence, "owner"),
		Gap: id(artifact.KindEvidence, "gap"), Falsifier: id(artifact.KindRecipe, "falsifier"),
		GapState: trainingprogram.MechanismGapOpen,
	}
	license := id(artifact.KindEvidence, "license")
	specification, err := json.Marshal(provenanceSpecification{
		Assessments: []trainingprogram.MechanismAssessment{assessment}, Repository: "local/marin",
		Commit: "b352a9e5d49eede33f08e0ab8df1e786425c2db5", License: license,
		Sources: []runrecord.ExternalMechanismSource{{Mechanism: assessment.Mechanism, Paths: []string{"experiments/grug/moe/model.py"}}},
	})
	testcheck.NoError(t, err)
	root := t.TempDir()
	input := filepath.Join(root, "provenance.json")
	testcheck.NoError(t, os.WriteFile(input, specification, 0o600))
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Create(storePath)
	testcheck.NoError(t, err)
	parents := []artifact.Descriptor{
		{ID: assessment.Mechanism}, {ID: assessment.Owner}, {ID: assessment.Gap}, {ID: assessment.Falsifier}, {ID: license},
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/external-provenance/parents", Artifacts: parents}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-provenance", input, "-store", storePath}, &output))
	if !strings.Contains(output.String(), "mechanisms=1 transfer=reexpress-overgo-native") {
		t.Fatalf("output = %q", output.String())
	}
}

func TestRunPublishesCompleteTypedMechanismEvidenceBundle(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	mechanism := id(artifact.KindRecipe, "batched-evidence-mechanism")
	sources := []mechanismEvidenceSource{
		{Role: trainingprogram.MechanismEvidenceGap, Source: id(artifact.KindEvidence, "gap-source")},
		{Role: trainingprogram.MechanismEvidenceFalsifier, Source: id(artifact.KindRecipe, "falsifier-source")},
		{Role: trainingprogram.MechanismEvidenceCostBound, Source: id(artifact.KindEvidence, "cost-source")},
		{Role: trainingprogram.MechanismEvidenceBenefit, Source: id(artifact.KindEvidence, "benefit-source")},
	}
	data, err := json.Marshal(evidenceSpecification{Mechanism: mechanism, Sources: sources})
	testcheck.NoError(t, err)
	root := t.TempDir()
	input := filepath.Join(root, "evidence.json")
	testcheck.NoError(t, os.WriteFile(input, data, 0o600))
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Create(storePath)
	testcheck.NoError(t, err)
	descriptors := []artifact.Descriptor{{ID: mechanism}}
	for _, source := range sources {
		descriptors = append(descriptors, artifact.Descriptor{ID: source.Source})
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/mechanism-evidence/parents", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-evidence", input, "-store", storePath}, &output))
	if !strings.Contains(output.String(), "claims=4") {
		t.Fatalf("output = %q", output.String())
	}
	store = overgodbtest.OpenReadOnly(t, storePath)
	for _, source := range sources {
		want, err := trainingprogram.NewMechanismEvidence(source.Role, mechanism, source.Source)
		testcheck.NoError(t, err)
		got, err := trainingprogram.RequireMechanismEvidence(t.Context(), store, want.ID, source.Role)
		if err != nil || got != want {
			t.Fatalf("evidence %s = (%+v, %v), want %+v", source.Role, got, err, want)
		}
	}
}
