package artifact

import (
	"encoding/json"
	"slices"
	"testing"
)

type codecFixture struct {
	Names []string `json:"names"`
	ID    ID       `json:"-"`
}

func TestDocumentCodecOwnsCanonicalLifecycle(t *testing.T) {
	contract := DocumentContract{
		Kind: KindEvidence, MediaType: "application/vnd.overgo.codec-fixture+json",
		Schema: "overgo/codec-fixture/v1",
	}
	codec := DocumentCodec[codecFixture]{
		Name: "codec fixture", Contract: contract,
		Decode: func(data []byte, value *codecFixture) error { return json.Unmarshal(data, value) },
		Encode: func(value codecFixture) ([]byte, error) { return json.Marshal(value) },
		Canonicalize: func(value *codecFixture) error {
			slices.Sort(value.Names)
			return nil
		},
		Clone: func(value codecFixture) codecFixture {
			value.Names = slices.Clone(value.Names)
			return value
		},
		Identity:    func(value codecFixture) ID { return value.ID },
		SetIdentity: func(value *codecFixture, id ID) { value.ID = id },
	}
	input := codecFixture{Names: []string{"second", "first"}}
	document, err := codec.New(input)
	if err != nil {
		t.Fatal(err)
	}
	if input.Names[0] != "second" || !slices.Equal(document.Names, []string{"first", "second"}) {
		t.Fatalf("input/document names = %v/%v", input.Names, document.Names)
	}
	content, err := codec.Content(document)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := codec.Parse(content.Data)
	if err != nil || parsed.ID != document.ID || !slices.Equal(parsed.Names, document.Names) {
		t.Fatalf("parsed document = %+v/%v", parsed, err)
	}
	mutated := document
	mutated.Names = []string{"second", "first"}
	if err := codec.ValidateIdentity(mutated); err == nil {
		t.Fatal("non-canonical document accepted")
	}
	if _, err := codec.Parse([]byte(`{"names":["second","first"]}`)); err == nil {
		t.Fatal("non-canonical encoding accepted")
	}
}
