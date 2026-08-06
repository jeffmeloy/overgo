package artifact

import "testing"

func TestDocumentContractBuildsAndValidatesContent(t *testing.T) {
	contract := DocumentContract{
		Kind: KindDataset, MediaType: "application/test+json", Schema: "test/v1",
	}
	data := []byte(`{"version":1}`)
	id, err := contract.Identify(data)
	if err != nil {
		t.Fatal(err)
	}
	content, err := contract.Content(id, data)
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 'x'
	if content.Data[0] != '{' {
		t.Fatal("content retained caller storage")
	}
	if err := contract.ValidateContent(content, id); err != nil {
		t.Fatal(err)
	}
	other := contract
	other.Schema = "test/v2"
	if err := other.ValidateContent(content, id); err == nil {
		t.Fatal("incompatible schema accepted")
	}
}

func TestDocumentContractRejectsInvalidIdentityAndBounds(t *testing.T) {
	contract := DocumentContract{Kind: KindRun, MediaType: "application/test+json", Schema: "test/v1"}
	data := []byte(`{"version":1}`)
	wrong, err := IdentifyBytes(KindRun, []byte("wrong"))
	if err != nil {
		t.Fatal(err)
	}
	if err := contract.ValidateIdentity(wrong, data); err == nil {
		t.Fatal("wrong identity accepted")
	}
	if _, err := contract.Identify(nil); err == nil {
		t.Fatal("empty document accepted")
	}
	invalid := contract
	invalid.Kind = KindInvalid
	if _, err := invalid.Identify(data); err == nil {
		t.Fatal("invalid kind accepted")
	}
}
