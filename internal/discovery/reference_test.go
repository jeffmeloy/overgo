package discovery

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMatchReference(t *testing.T) {
	first := CatalogEntry{Model: testutil.ArtifactID(t, artifact.KindModel, "first"), Location: "/first/shared.gguf", Present: true}
	second := CatalogEntry{Model: testutil.ArtifactID(t, artifact.KindModel, "second"), Location: "/second/shared.gguf", Present: true}
	// Put an exact basename-shaped location after two ambiguous aliases: exact
	// selection must not depend on enumeration order.
	exact := CatalogEntry{Model: testutil.ArtifactID(t, artifact.KindModel, "exact"), Location: "shared", Present: true}
	for _, test := range []struct {
		name      string
		entries   []CatalogEntry
		truncated bool
		reference string
		want      artifact.ID
		refuse    bool
	}{
		{name: "identity", entries: []CatalogEntry{first, second}, reference: second.Model.String(), want: second.Model},
		{name: "location", entries: []CatalogEntry{first, second}, reference: first.Location, want: first.Model},
		{name: "alias", entries: []CatalogEntry{first}, reference: "SHARED", want: first.Model},
		{name: "ambiguous", entries: []CatalogEntry{first, second}, reference: "shared", refuse: true},
		{name: "exact-priority", entries: []CatalogEntry{first, second, exact}, reference: "shared", want: exact.Model},
		{name: "missing", entries: []CatalogEntry{first}, reference: "absent"},
		{name: "truncated", entries: []CatalogEntry{first}, truncated: true, reference: first.Model.String(), refuse: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			entry, found, err := MatchReference(test.entries, test.truncated, test.reference)
			if (err != nil) != test.refuse || found != test.want.Valid() || found && entry.Model != test.want {
				t.Fatalf("match=%s found=%t error=%v want=%s refusal=%t", entry.Model, found, err, test.want, test.refuse)
			}
		})
	}
}
