package remoteprovider

import (
	"strings"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestAProviderBindsOnlyAVariableTheOperatorNamed holds the declaration's two rules (serving-bounds/provider-variables):
// a variable is bound only once the operator named it, a server secret is never named, and naming one twice changes
// nothing; a key goes over https or over http to this machine alone, at declaration and again when the key is read, so
// a provider declared at an http endpoint elsewhere before the rule is refused its key.
func TestAProviderBindsOnlyAVariableTheOperatorNamed(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	document := Document{Name: "named", Endpoint: "https://provider.example/api/v1", KeyEnvironment: "PROVIDER_NAMED_KEY", Models: []string{"vendor/one"}}
	if _, err := DeclareDocument(ctx, store, documentModelLimit, document, setKeyTestCommit); err == nil || !strings.Contains(err.Error(), "PROVIDER_NAMED_KEY is not a variable the operator named") {
		t.Fatalf("the declaration of an unnamed variable = %v, want it refused by name", err)
	}
	if err := NameVariable(ctx, store, "OVERGO_API_KEY"); err == nil {
		t.Fatal("the server's own key was named for providers")
	}
	for range 2 {
		testcheck.NoError(t, NameVariable(ctx, store, " PROVIDER_NAMED_KEY "))
	}
	if named := testcheck.Must(namedVariables(ctx, store))(t); len(named) != 1 || named[0] != "PROVIDER_NAMED_KEY" {
		t.Fatalf("the named variables are %q, want the one named twice", named)
	}
	clear := document
	clear.Endpoint = "http://provider.example/api/v1"
	if _, err := DeclareDocument(ctx, store, documentModelLimit, clear, setKeyTestCommit); err == nil || !strings.Contains(err.Error(), "only over https") {
		t.Fatalf("a declaration at an http endpoint elsewhere = %v, want it refused", err)
	}
	if declared, err := DeclareDocument(ctx, store, documentModelLimit, document, setKeyTestCommit); err != nil || len(declared) != 1 {
		t.Fatalf("the named variable's declaration = %+v, %v", declared, err)
	}
	for endpoint, admitted := range map[string]bool{
		"https://provider.example/api/v1": true, "http://localhost:8080/v1": true, "http://127.0.0.1:9/v1": true, "http://[::1]:9/v1": true,
		"http://provider.example/api/v1": false, "http://10.0.0.5/v1": false, "ftp://provider.example": false,
	} {
		if err := KeyTransport(endpoint); (err == nil) != admitted {
			t.Errorf("a key to %s is admitted %t, want %t: %v", endpoint, err == nil, admitted, err)
		}
	}
	t.Setenv("PROVIDER_EARLIER_KEY", "earlier-secret")
	if _, err := Key(Provider{Endpoint: "http://provider.example/api/v1", KeyEnvironment: "PROVIDER_EARLIER_KEY"}); err == nil {
		t.Fatal("a key was read for an http endpoint elsewhere")
	}
}
