package remoterelay

import (
	"strings"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/testcheck"
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

// TestAProviderBindsOnlyAVariableTheOperatorNamed holds a listing to the variables the operator named
// (serving-bounds/provider-variables): a declaration made before the rule binds a variable to this machine's endpoint,
// yet a listing by it is refused by the variable's name until the operator names it, and then answers under its key; a
// key typed for an http endpoint elsewhere is refused before anything is sent.
//
// Serial: it sets the key's variable.
func TestAProviderBindsOnlyAVariableTheOperatorNamed(t *testing.T) {
	const variable = "RELAY_NAMED_KEY"
	ctx := t.Context()
	t.Setenv(variable, "named-key")
	fake, _ := relaytest.ServeListing(t, "", "named-key", []string{"x"}, []relaytest.Model{{ID: "vendor/one"}})
	store := overgodbtest.Open(t, t.TempDir())
	if _, err := remoteprovider.Declare(ctx, store, remoteprovider.Provider{Name: "earlier", Endpoint: fake.URL, KeyEnvironment: variable, Model: "vendor/one"},
		"0123456789abcdef0123456789abcdef01234567"); err != nil {
		t.Fatal(err)
	}
	listing := Listing{Endpoint: fake.URL, KeyEnvironment: variable}
	if _, err := ListModels(ctx, store, 8, listing, fake.Client()); err == nil || !strings.Contains(err.Error(), variable+" is not a variable the operator named") {
		t.Fatalf("a listing by an unnamed variable = %v, want it refused by name", err)
	}
	testcheck.NoError(t, remoteprovider.NameVariable(ctx, store, variable))
	if models, err := ListModels(ctx, store, 8, listing, fake.Client()); err != nil || len(models) != 1 {
		t.Fatalf("the named variable's listing = %+v, %v", models, err)
	}
	if _, err := ListModels(ctx, nil, 0, Listing{Endpoint: "http://provider.example/api/v1", Key: "typed-key"}, fake.Client()); err == nil || !strings.Contains(err.Error(), "only over https") {
		t.Fatalf("a typed key for an http endpoint elsewhere = %v, want it refused", err)
	}
}
