package modelrecipe

import (
	"testing"

	"overgo/internal/model"
)

// TestCanonicalDocumentPublicationOwners proves this package's typed
// documents publish only through their codec owners: the profile
// document's content bytes validate its bound identity and parse
// back to the identical document.
func TestCanonicalDocumentPublicationOwners(t *testing.T) {
	profile, ok := model.LookupArchitecture("llama")
	if !ok {
		t.Fatal("llama architecture profile unavailable")
	}
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	if err := document.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseProfileDocument(content)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed.ID, err)
	}
}
