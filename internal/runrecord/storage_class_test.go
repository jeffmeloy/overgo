package runrecord

import (
	"testing"

	"overgo/internal/overgodb"
)

// TestRSIStorageClassContract proves this package's document families
// respect the ownership rules: none of the typed schemas it publishes
// is shaped like an operational signal.
func TestRSIStorageClassContract(t *testing.T) {
	for _, schema := range []string{
		RunSchema, EvaluationSchema, interactionTraceSchema,
		efficiencyTraceSchema, BudgetSchema, BudgetChargeSchema,
	} {
		if overgodb.IsOperationalSignalSchema(schema) {
			t.Fatalf("schema %q is shaped like an operational signal", schema)
		}
	}
}
