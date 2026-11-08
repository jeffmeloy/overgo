package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestAQualityCellMeasuredOnAnEarlierDefinitionIsOwedOnTheServedOne holds the coverage to which modality-quality results
// hold on the definition a model serves (SQA-277): six E4B quality cells an external adapter measured on an earlier
// model definition read as current acceptance beside the 24 cells re-run on the served one. A quality cell from an
// earlier definition is projected owed on the served one, both named; one from the served definition, a protocol cell,
// and a cell whose model serves no known definition are not.
func TestAQualityCellMeasuredOnAnEarlierDefinitionIsOwedOnTheServedOne(t *testing.T) {
	t.Parallel()
	served := testutil.ArtifactID(t, artifact.KindModelDefinition, "the served definition")
	earlier := testutil.ArtifactID(t, artifact.KindModelDefinition, "an earlier definition")
	gap := owedQuality(modelValidationCell{Name: "quality/image/native", ModelDefinition: earlier}, served)
	if !strings.Contains(gap, served.String()) || !strings.Contains(gap, earlier.String()) || !strings.Contains(gap, "Owed") {
		t.Fatalf("a quality cell from an earlier definition reads %q; want it owed on the served one, both named", gap)
	}
	for name, cell := range map[string]modelValidationCell{
		"on the served definition": {Name: "quality/image/native", ModelDefinition: served},
		"a protocol cell":          {Name: "protocol/image/native", ModelDefinition: earlier},
	} {
		if gap := owedQuality(cell, served); gap != "" {
			t.Errorf("a cell %s reads owed: %q", name, gap)
		}
	}
	if gap := owedQuality(modelValidationCell{Name: "quality/image/native", ModelDefinition: earlier}, artifact.ID{}); gap != "" {
		t.Errorf("a cell whose model serves no known definition reads owed: %q", gap)
	}
}
