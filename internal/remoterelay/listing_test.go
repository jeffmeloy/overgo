package remoterelay

import (
	"strings"
	"testing"

	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay/relaytest"
)

// TestListModelsReadsTheProviderListing: the listing comes back with ids,
// names and context lengths, blank ids dropped; the key's absence and a
// provider without the route are refused with their reasons.
func TestListModelsReadsTheProviderListing(t *testing.T) {
	fake, _ := relaytest.ServeListing(t, "", "listing-key", []string{"x"}, []relaytest.Model{
		{ID: "vendor/one", Name: "One", ContextLength: 4096}, {ID: "vendor/two"}, {ID: " "},
	})
	t.Setenv("OVERGO_LISTING_TEST_KEY", "listing-key")
	provider := remoteprovider.Provider{Endpoint: fake.URL, KeyEnvironment: "OVERGO_LISTING_TEST_KEY"}
	models, err := ListModels(t.Context(), provider, fake.Client())
	if err != nil || len(models) != 2 || models[0] != (Listed{ID: "vendor/one", Name: "One", ContextLength: 4096}) || models[1].ID != "vendor/two" {
		t.Fatalf("listed = %+v, %v", models, err)
	}
	bare, _ := relaytest.Serve(t, "", "listing-key", []string{"x"})
	if _, err := ListModels(t.Context(), remoteprovider.Provider{Endpoint: bare.URL, KeyEnvironment: "OVERGO_LISTING_TEST_KEY"}, bare.Client()); err == nil ||
		!strings.Contains(err.Error(), "answered 404") {
		t.Fatalf("provider without a listing: %v", err)
	}
	t.Setenv("OVERGO_LISTING_TEST_KEY", "")
	if _, err := ListModels(t.Context(), provider, fake.Client()); err == nil || !strings.Contains(err.Error(), "OVERGO_LISTING_TEST_KEY is not set") {
		t.Fatalf("keyless listing: %v", err)
	}
}
