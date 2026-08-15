package artifact

import (
	"context"
	"encoding/json"
	"slices"
	"testing"
)

type codecFixture struct {
	Names []string `json:"names"`
	ID    ID       `json:"-"`
}

type absentDocumentReader struct{ Reader }

func (absentDocumentReader) Content(context.Context, ID) (Content, bool, error) {
	return Content{}, false, nil
}

func TestDocumentCodecOwnsCanonicalLifecycle(t *testing.T) {
	const expectedEncodesPerOperation = 1
	contract := DocumentContract{
		Kind: KindEvidence, MediaType: "application/vnd.overgo.codec-fixture+json",
		Schema: "overgo/codec-fixture/v1",
	}
	encodeCalls := 0
	codec := DocumentCodec[codecFixture]{
		Name: "codec fixture", Contract: contract,
		Decode: func(data []byte, value *codecFixture) error { return json.Unmarshal(data, value) },
		Encode: func(value codecFixture) ([]byte, error) {
			encodeCalls++
			return json.Marshal(value)
		},
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
	encodeCalls = 0
	content, err := codec.Content(document)
	if err != nil {
		t.Fatal(err)
	}
	if encodeCalls != expectedEncodesPerOperation {
		t.Fatalf("content encoded %d times", encodeCalls)
	}
	encodeCalls = 0
	parsed, err := codec.Parse(content.Data)
	if err != nil || parsed.ID != document.ID || !slices.Equal(parsed.Names, document.Names) {
		t.Fatalf("parsed document = %+v/%v", parsed, err)
	}
	if encodeCalls != expectedEncodesPerOperation {
		t.Fatalf("parse encoded %d times", encodeCalls)
	}
	loaded, ok, err := codec.Read(context.Background(), documentReader{content: content}, document.ID)
	if err != nil || !ok || loaded.ID != document.ID || !slices.Equal(loaded.Names, document.Names) {
		t.Fatalf("read document = %+v/%v/%v", loaded, ok, err)
	}
	required, err := codec.Require(context.Background(), documentReader{content: content}, document.ID)
	if err != nil || required.ID != document.ID {
		t.Fatalf("required document = %+v/%v", required, err)
	}
	if _, err := codec.Require(context.Background(), absentDocumentReader{}, document.ID); err == nil {
		t.Fatal("absent required document accepted")
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
