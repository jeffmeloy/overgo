package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestRSIStorageClassContract proves this package's document families
// respect the ownership rules: every typed schema it publishes
// classifies as a canonical fact -- transcripts and observation
// chunks carry their bulk as blob-backed content bytes, never as a
// separate log -- and none of them is shaped like an operational
// signal.
func TestRSIStorageClassContract(t *testing.T) {
	for _, schema := range []string{
		RunSchema, EvaluationSchema, interactionTraceSchema,
		EfficiencyTraceSchema, BudgetSchema, BudgetChargeSchema,
	} {
		descriptor := artifact.Descriptor{Schema: schema, MediaType: "application/json"}
		if class := overgodb.DocumentStorageClass(descriptor); class != overgodb.ClassCanonicalFact {
			t.Fatalf("schema %q classifies %s, want canonical fact", schema, class)
		}
		if overgodb.IsOperationalSignalSchema(schema) {
			t.Fatalf("schema %q is shaped like an operational signal", schema)
		}
	}
}
