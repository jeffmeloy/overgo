package modelrecipe

import (
	"testing"

	"overgo/internal/model"
	"overgo/internal/testcheck"
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
	testcheck.NoError(t, err)
	testcheck.NoError(t, document.ValidateIdentity())
	content, err := document.Content()
	testcheck.NoError(t, err)
	parsed, err := ParseProfileDocument(content)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed.ID, err)
	}
}
