package gate

import (
	"slices"
	"testing"
)

// TestGateReadsPublishedDocuments requires the pipeline to carry the check
// that walks published OvergoDB documents through the production readers. Every
// other check compares source against source; this declaration is what makes
// a schema change against already-published documents visible to the gate at
// all, so its ownership must cover the document-schema owners and its trigger
// must be its own ownership fact.
func TestGateReadsPublishedDocuments(t *testing.T) {
	t.Parallel()
	for _, check := range (&gateContext{}).pipelineChecks() {
		if check.Descriptor.Name != "published" {
			continue
		}
		ownership := check.Descriptor.Ownership
		if ownership.Fact == "" || !slices.Contains(check.Descriptor.Triggers, ownership.Fact) {
			t.Fatalf("published check trigger and ownership disagree: %+v", check.Descriptor)
		}
		for _, owner := range []string{"internal/modelrecipe", "internal/recipe", "internal/artifact"} {
			if !slices.Contains(ownership.Packages, owner) {
				t.Fatalf("published check does not own %s: %+v", owner, ownership.Packages)
			}
		}
		if check.Descriptor.Always {
			t.Fatal("published check must select by ownership, not run always")
		}
		return
	}
	t.Fatal("gate pipeline carries no published-documents check")
}
