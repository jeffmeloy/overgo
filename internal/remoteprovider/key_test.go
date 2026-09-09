package remoteprovider

import (
	"os"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

const setKeyTestCommit = "0123456789abcdef0123456789abcdef01234567"

// TestSetKeyPlacesTheKeyInTheEnvironment: the key lands in the declared
// variable for this process (the refusal lifts), by location or model
// identity; an undeclared reference and an empty key are refused.
func TestSetKeyPlacesTheKeyInTheEnvironment(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	t.Setenv("OVERGO_SET_KEY_TEST", "")
	declaration, err := Declare(t.Context(), store, Provider{
		Name: "fake", Endpoint: "https://fake.example/api/v1", KeyEnvironment: "OVERGO_SET_KEY_TEST", Model: "vendor/model",
	}, setKeyTestCommit)
	if err != nil {
		t.Fatal(err)
	}
	if Refusal(declaration.Provider) == "" {
		t.Fatal("a keyless declaration is not refused")
	}
	provider, err := SetKey(t.Context(), store, 16, declaration.Location, "secret")
	if err != nil || provider.KeyEnvironment != "OVERGO_SET_KEY_TEST" || os.Getenv("OVERGO_SET_KEY_TEST") != "secret" || Refusal(provider) != "" {
		t.Fatalf("set key = %+v, %v; environment %q", provider, err, os.Getenv("OVERGO_SET_KEY_TEST"))
	}
	if _, err := SetKey(t.Context(), store, 16, declaration.Model.String(), "other"); err != nil || os.Getenv("OVERGO_SET_KEY_TEST") != "other" {
		t.Fatalf("set key by identity: %v; environment %q", err, os.Getenv("OVERGO_SET_KEY_TEST"))
	}
	if _, err := SetKey(t.Context(), store, 16, "remote://nobody/model", "k"); err == nil || !strings.Contains(err.Error(), "names no declared model") {
		t.Fatalf("undeclared reference: %v", err)
	}
	if _, err := SetKey(t.Context(), store, 16, declaration.Location, " "); err == nil || !strings.Contains(err.Error(), "key is empty") {
		t.Fatalf("empty key: %v", err)
	}
}
