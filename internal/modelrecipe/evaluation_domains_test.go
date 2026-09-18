package modelrecipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// These IDs were retained from the evaluation-owned codec before relocation.
// Reading metadata must not require linking model execution or change history.
func TestEvaluationDomainIdentity(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "domain-model")
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/domain/model", Artifacts: []artifact.Descriptor{{ID: model}}}); err != nil {
		t.Fatal(err)
	}
	if _, declared, err := EvalDomains(t.Context(), store, model); err != nil || declared {
		t.Fatalf("undeclared=%v error=%v", declared, err)
	}
	for _, test := range []struct {
		input, want []string
		id          string
	}{
		{[]string{"dna"}, []string{"dna"}, "evidence:sha256:d5d84b61cb3a770e3c951eaa3b9a4632c4ca3e09c2ed5cd88c885cf88c1e2193"},
		{[]string{"text", "dna"}, []string{"dna", "text"}, "evidence:sha256:15f727d9ba07dbea0ef813ec6e006a45e592cd24f1f440842d6eb39f6f7acf5a"},
	} {
		before := slices.Clone(test.input)
		id, err := DeclareEvalDomains(t.Context(), store, model, test.input)
		if err != nil || id.String() != test.id || !slices.Equal(before, test.input) {
			t.Fatalf("identity/input changed: %s %v", id, err)
		}
		domains, declared, err := EvalDomains(t.Context(), store, model)
		if err != nil || !declared || !slices.Equal(domains, test.want) {
			t.Fatalf("domains=%v declared=%v error=%v", domains, declared, err)
		}
		_, beforeSequence := store.Head()
		again, err := DeclareEvalDomains(t.Context(), store, model, test.input)
		_, afterSequence := store.Head()
		if err != nil || again != id || beforeSequence != afterSequence {
			t.Fatalf("repeat declaration changed store: %v", err)
		}
	}
	for _, invalid := range [][]string{nil, {""}, {" dna"}, {"dna", "dna"}} {
		if _, err := DeclareEvalDomains(t.Context(), store, model, invalid); err == nil {
			t.Fatalf("accepted invalid domains %q", invalid)
		}
	}
}
