package remoteprovider

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

const testCodeCommit = "0123456789abcdef0123456789abcdef01234567"

func TestListPagesRetiredHistoryAndBoundsActiveModels(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	provider := Provider{Name: "history", KeyEnvironment: "HISTORY_KEY", Model: "model"}
	for _, endpoint := range []string{"https://old.example", "https://older.example"} {
		provider.Endpoint = endpoint
		declaration, err := Declare(t.Context(), store, provider, testCodeCommit)
		if err != nil {
			t.Fatal(err)
		}
		if err := Retire(t.Context(), store, 1, declaration.Location, testCodeCommit, "provider endpoint replaced"); err != nil {
			t.Fatal(err)
		}
	}
	provider.Endpoint = "https://current.example"
	current, err := Declare(t.Context(), store, provider, testCodeCommit)
	if err != nil {
		t.Fatal(err)
	}
	listed, err := List(t.Context(), store, 1)
	if err != nil || len(listed) != 1 || listed[0].Model != current.Model {
		t.Fatalf("active listing after retired history = %+v, %v", listed, err)
	}
	if _, _, remote, err := Reference(t.Context(), store, 1, "local.gguf"); err != nil || remote {
		t.Fatalf("local model lookup after retired history: remote=%v, error=%v", remote, err)
	}
	provider.Model = "another"
	if _, err := Declare(t.Context(), store, provider, testCodeCommit); err != nil {
		t.Fatal(err)
	}
	if _, err := List(t.Context(), store, 1); err == nil || !strings.Contains(err.Error(), "active model catalog exceeds") {
		t.Fatalf("active listing overflow = %v", err)
	}
}

// TestDeclareListsTheRemoteModelWithItsRefusal pins the declaration end to
// end: the provider document, the model manifest at its remote location
// and the activated remote inference recipe enter the store; the servable
// catalog lists the model at that location, refused by the key's variable
// name while the environment lacks it and servable once it holds it; the
// serving reference resolves by location and by model identity; the
// declaration's environment is not reproducible; and declaring again is a
// no-op returning the same identities.
func TestDeclareListsTheRemoteModelWithItsRefusal(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	t.Setenv("OVERGO_REMOTE_PROVIDER_TEST_KEY", "")
	provider := Provider{
		Name: "fake", Endpoint: "https://fake.example/api/v1/", KeyEnvironment: "OVERGO_REMOTE_PROVIDER_TEST_KEY", Model: "vendor/model",
	}
	declaration, err := Declare(t.Context(), store, provider, testCodeCommit)
	if err != nil {
		t.Fatal(err)
	}
	if declaration.Location != "remote://fake/vendor/model" || declaration.Model.Kind() != artifact.KindModel ||
		declaration.Recipe.Task != recipe.TaskInference || declaration.Provider.Endpoint != "https://fake.example/api/v1" {
		t.Fatalf("declaration = %+v", declaration)
	}
	activation, active, err := modelrecipe.ActiveRecord(t.Context(), store, declaration.Model, recipe.TaskInference)
	if err != nil || !active || activation.Definition.ID != declaration.Recipe.ID || activation.Tier != recipe.EvidenceExperimental {
		t.Fatalf("activation = %+v active=%v err=%v", activation, active, err)
	}
	if refusal := Refusal(declaration.Provider); !strings.Contains(refusal, "OVERGO_REMOTE_PROVIDER_TEST_KEY is not set") {
		t.Fatalf("keyless refusal = %q", refusal)
	}
	t.Setenv("OVERGO_REMOTE_PROVIDER_TEST_KEY", "secret")
	if refusal := Refusal(declaration.Provider); refusal != "" {
		t.Fatalf("keyed refusal = %q", refusal)
	}
	listed, err := List(t.Context(), store, 16)
	if err != nil || len(listed) != 1 || listed[0].Model != declaration.Model || listed[0].Location != declaration.Location {
		t.Fatalf("listed = %+v, %v", listed, err)
	}
	for _, reference := range []string{declaration.Location, declaration.Model.String()} {
		resolved, modelID, remote, err := Reference(t.Context(), store, 16, reference)
		if err != nil || !remote || modelID != declaration.Model || resolved.ID != declaration.Provider.ID {
			t.Fatalf("reference %q = %+v %s remote=%v err=%v", reference, resolved, modelID, remote, err)
		}
	}
	if _, _, remote, err := Reference(t.Context(), store, 16, "C:/models/local.gguf"); err != nil || remote {
		t.Fatalf("a local path resolved remotely: %v, %v", remote, err)
	}
	environment, err := Environment(declaration.Provider)
	if err != nil || environment.Reproducible() || environment.Host != "fake.example" || environment.Device != "vendor/model" {
		t.Fatalf("environment = %+v, %v", environment, err)
	}
	again, err := Declare(t.Context(), store, provider, testCodeCommit)
	if err != nil || again.Model != declaration.Model || again.Recipe.ID != declaration.Recipe.ID {
		t.Fatalf("second declaration = %+v, %v", again, err)
	}
}

// TestProviderValidation pins the declaration's refusals: a name with a
// slash, an endpoint that is not an http(s) URL, and a missing key
// variable or model id.
func TestProviderValidation(t *testing.T) {
	valid := Provider{Name: "fake", Endpoint: "https://fake.example/v1", KeyEnvironment: "K", Model: "m"}
	if _, err := New(valid); err != nil {
		t.Fatal(err)
	}
	for name, invalid := range map[string]Provider{
		"slashed name":  {Name: "a/b", Endpoint: valid.Endpoint, KeyEnvironment: "K", Model: "m"},
		"bare endpoint": {Name: "fake", Endpoint: "fake.example", KeyEnvironment: "K", Model: "m"},
		"no key":        {Name: "fake", Endpoint: valid.Endpoint, Model: "m"},
		"no model":      {Name: "fake", Endpoint: valid.Endpoint, KeyEnvironment: "K"},
	} {
		if _, err := New(invalid); err == nil {
			t.Errorf("%s validated", name)
		}
	}
}
