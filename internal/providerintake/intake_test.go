package providerintake

import (
	"strings"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/server"
	"overgo/internal/testcheck"
)

const (
	intakeTestVariable = "OVERGO_PROVIDER_INTAKE_TEST_KEY"
	intakeTestCommit   = "0123456789abcdef0123456789abcdef01234567"
)

// TestIntakeListsProviderModels pins the listing intake: the provider's own
// listing under a typed key or a declared variable, each model with its
// declared context length; an undeclared variable and an absent key refuse.
func TestIntakeListsProviderModels(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	fake, _ := relaytest.ServeListing(t, "", "intake-key", []string{"unused"}, []relaytest.Model{{ID: "vendor/listed", Name: "Listed", ContextLength: 2048}})
	intake := Intake{CatalogLimit: 8}
	declared := server.ProviderListing{Endpoint: fake.URL + "/", KeyEnvironment: intakeTestVariable}
	t.Setenv(intakeTestVariable, "")
	if _, err := intake.ListModels(t.Context(), store, declared); err == nil {
		t.Fatal("a listing under an undeclared variable succeeded")
	}
	if _, err := remoteprovider.Declare(t.Context(), store, remoteprovider.Provider{
		Name: "intake", Endpoint: fake.URL, KeyEnvironment: intakeTestVariable, Model: "vendor/listed",
	}, intakeTestCommit); err != nil {
		t.Fatal(err)
	}
	if _, err := intake.ListModels(t.Context(), store, declared); err == nil || !strings.Contains(err.Error(), intakeTestVariable+" is not set") {
		t.Fatalf("a listing without the key: %v", err)
	}
	t.Setenv(intakeTestVariable, "intake-key")
	for _, listing := range []server.ProviderListing{declared, {Endpoint: fake.URL, Key: "intake-key"}} {
		models, err := intake.ListModels(t.Context(), store, listing)
		testcheck.NoError(t, err)
		if len(models) != 1 || models[0].ID != "vendor/listed" || models[0].Name != "Listed" || models[0].ContextLength != 2048 {
			t.Fatalf("listed = %+v", models)
		}
	}
}
