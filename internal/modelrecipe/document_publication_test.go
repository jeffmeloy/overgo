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
	document := testcheck.Must(NewProfileDocument(profile))(t)
	testcheck.NoError(t, document.ValidateIdentity())
	content := testcheck.Must(document.Content())(t)
	parsed, err := ParseProfileDocument(content)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed.ID, err)
	}
}
