package remoterelay

import (
	"strings"
	"testing"

	"overgo/internal/remoterelay/relaytest"
)

// TestListModelsReadsTheProviderListing: the listing comes back with ids,
// names and context lengths, blank ids dropped; a provider without the
// route is refused with its reason.
func TestListModelsReadsTheProviderListing(t *testing.T) {
	fake, _ := relaytest.ServeListing(t, "", "listing-key", []string{"x"}, []relaytest.Model{
		{ID: "vendor/one", Name: "One", ContextLength: 4096}, {ID: "vendor/two"}, {ID: " "},
	})
	models, err := ListModels(t.Context(), nil, 0, Listing{Endpoint: fake.URL + "/", Key: "listing-key"}, fake.Client())
	if err != nil || len(models) != 2 || models[0] != (Listed{ID: "vendor/one", Name: "One", ContextLength: 4096}) || models[1].ID != "vendor/two" {
		t.Fatalf("listed = %+v, %v", models, err)
	}
	bare, _ := relaytest.Serve(t, "", "listing-key", []string{"x"})
	if _, err := ListModels(t.Context(), nil, 0, Listing{Endpoint: bare.URL, Key: "listing-key"}, bare.Client()); err == nil ||
		!strings.Contains(err.Error(), "answered 404") {
		t.Fatalf("provider without a listing: %v", err)
	}
}
