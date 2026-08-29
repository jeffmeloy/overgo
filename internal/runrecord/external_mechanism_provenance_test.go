package runrecord

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestExternalMechanismProvenanceIsExactAndNoRuntimeDependencyIsAdmitted(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "quantile-router"), Owner: id(artifact.KindEvidence, "router-owner"),
		Gap: id(artifact.KindEvidence, "router-gap"), Falsifier: id(artifact.KindRecipe, "router-falsifier"),
		GapState: trainingprogram.MechanismGapOpen,
	}
	census, err := trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment})
	if err != nil {
		t.Fatal(err)
	}
	source := ExternalMechanismSource{Mechanism: assessment.Mechanism, Paths: []string{
		"experiments/grug/moe/model.py", "experiments/grug/moe/README.md",
	}}
	license := id(artifact.KindEvidence, "apache-2.0-license")
	provenance, err := NewExternalMechanismProvenance(
		census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license, []ExternalMechanismSource{source},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := provenance.Content()
	if err != nil || provenance.Transfer != externalMechanismTransfer || !bytes.Contains(content.Data, []byte(`"transfer":"reexpress-overgo-native"`)) {
		t.Fatalf("provenance = (%q, %v)", provenance.Transfer, err)
	}
	for _, forbidden := range [][]byte{[]byte(`"runtime_dependencies"`), []byte(`"source_imports"`), []byte(`"schema_imports"`)} {
		forged := bytes.Replace(content.Data, []byte(`"sources":`), append(forbidden, []byte(`:["foreign"],"sources":`)...), 1)
		if _, err := externalMechanismProvenanceCodec.Parse(forged); err == nil {
			t.Fatalf("forbidden dependency field %s accepted", forbidden)
		}
	}
	parents := make(map[artifact.ID]bool)
	for _, edge := range provenance.Lineage() {
		parents[edge.Parent] = true
	}
	for _, required := range []artifact.ID{census.ID(), license, assessment.Mechanism} {
		if !parents[required] {
			t.Fatalf("provenance lineage omits %s", required)
		}
	}

	if _, err := NewExternalMechanismProvenance(census, "local/marin", "not-a-commit", license, []ExternalMechanismSource{source}); err == nil {
		t.Fatal("non-hex commit accepted")
	}
	foreign := source
	foreign.Mechanism = id(artifact.KindRecipe, "foreign-mechanism")
	if _, err := NewExternalMechanismProvenance(census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license, []ExternalMechanismSource{foreign}); err == nil {
		t.Fatal("mechanism outside census accepted")
	}
	missing := source
	missing.Paths = nil
	if _, err := NewExternalMechanismProvenance(census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license, []ExternalMechanismSource{missing}); err == nil {
		t.Fatal("source without exact paths accepted")
	}
	traversal := source
	traversal.Paths = []string{"../foreign/runtime.py"}
	if _, err := NewExternalMechanismProvenance(census, "local/marin", "b352a9e5d49eede33f08e0ab8df1e786425c2db5", license, []ExternalMechanismSource{traversal}); err == nil {
		t.Fatal("source path traversal accepted")
	}
}
