package providerintake

import (
	"testing"

	"overgo/internal/remoterelay/relaytest"
)

// TestIntakeListsProviderModels pins the listing intake: the provider's
// own listing under the key its variable holds, each model with its
// declared context length.
func TestIntakeListsProviderModels(t *testing.T) {
	t.Setenv("OVERGO_PROVIDER_INTAKE_TEST_KEY", "intake-key")
	fake, _ := relaytest.ServeListing(t, "", "intake-key", []string{"unused"}, []relaytest.Model{{ID: "vendor/listed", Name: "Listed", ContextLength: 2048}})
	models, err := Intake{CatalogLimit: 8}.ListModels(t.Context(), fake.URL+"/", "OVERGO_PROVIDER_INTAKE_TEST_KEY")
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "vendor/listed" || models[0].Name != "Listed" || models[0].ContextLength != 2048 {
		t.Fatalf("listed = %+v", models)
	}
	if _, err := (Intake{}).ListModels(t.Context(), fake.URL, "OVERGO_PROVIDER_INTAKE_ABSENT_KEY"); err == nil {
		t.Fatal("a listing without the key succeeded")
	}
}
