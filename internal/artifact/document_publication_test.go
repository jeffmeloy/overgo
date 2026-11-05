package artifact

import (
	"encoding/json"
	"testing"

	"overgo/internal/testcheck"
)

// TestCanonicalDocumentPublicationOwners holds ContentJSON to the
// publication contract: it produces exactly the bytes-and-identity
// pair the manual marshal-then-Content path produced, validates the
// bound identity, and refuses a value whose canonical bytes do not
// hash to the claimed id -- so no family has a reason to copy the
// pair again.
func TestCanonicalDocumentPublicationOwners(t *testing.T) {
	contract := DocumentContract{Kind: KindEvidence, MediaType: "application/vnd.overgo.fixture+json", Schema: "overgo/fixture/v1"}
	value := struct {
		Name string `json:"name"`
	}{Name: "publication-owner"}
	data, err := json.Marshal(value)
	testcheck.NoError(t, err)
	id, err := contract.Identify(data)
	testcheck.NoError(t, err)
	owned, err := contract.ContentJSON(id, value)
	testcheck.NoError(t, err)
	manual, err := contract.Content(id, data)
	testcheck.NoError(t, err)
	if owned.Descriptor != manual.Descriptor || string(owned.Data) != string(manual.Data) {
		t.Fatalf("owner output differs from manual pair: %+v vs %+v", owned, manual)
	}
	other := struct {
		Name string `json:"name"`
	}{Name: "different-value"}
	if _, err := contract.ContentJSON(id, other); err == nil {
		t.Fatal("mismatched identity admitted")
	}
}
