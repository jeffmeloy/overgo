package artifact

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"testing"
)

type codecFixture struct {
	Names []string `json:"names"`
	ID    ID       `json:"-"`
}

type absentDocumentReader struct{ Reader }

func (absentDocumentReader) OpenContent(context.Context, ID) (Descriptor, io.Reader, bool, error) {
	return Descriptor{}, nil, false, nil
}

type codecAliasReader struct {
	documentReader
	alias ID
}

func (r codecAliasReader) ResolveAlias(context.Context, string) (ID, bool, error) {
	return r.alias, r.alias.Valid(), nil
}

type missingAliasTargetReader struct{ codecAliasReader }

func (missingAliasTargetReader) OpenContent(context.Context, ID) (Descriptor, io.Reader, bool, error) {
	return Descriptor{}, nil, false, nil
}

func TestDocumentCodecReadOwnsCanonicalLifecycle(t *testing.T) {
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
	loaded, ok, err := codec.Read(t.Context(), documentReader{content: content}, document.ID)
	if err != nil || !ok || loaded.ID != document.ID || !slices.Equal(loaded.Names, document.Names) {
		t.Fatalf("read document = %+v/%v/%v", loaded, ok, err)
	}
	required, err := codec.Require(t.Context(), documentReader{content: content}, document.ID)
	if err != nil || required.ID != document.ID {
		t.Fatalf("required document = %+v/%v", required, err)
	}
	verified, err := codec.RequireVerified(
		t.Context(), documentReader{content: content}, document.ID,
		func(_ context.Context, _ Reader, value codecFixture) error {
			if value.ID != document.ID {
				return errors.New("unexpected verified document")
			}
			return nil
		},
	)
	if err != nil || verified.ID != document.ID {
		t.Fatalf("verified document = %+v/%v", verified, err)
	}
	verified, err = codec.RequireVerified(
		t.Context(), documentReader{content: content}, document.ID,
		func(_ context.Context, _ Reader, value codecFixture) error {
			value.Names[0] = "mutated"
			return nil
		},
	)
	if err != nil || !slices.Equal(verified.Names, document.Names) {
		t.Fatalf("verifier mutated returned document = %+v/%v", verified, err)
	}
	verified, err = codec.RequireVerified(
		t.Context(), documentReader{content: content}, document.ID,
		func(context.Context, Reader, codecFixture) error { return errors.New("rejected") },
	)
	if err == nil || verified.ID.Valid() || verified.Names != nil {
		t.Fatalf("rejected verified document = %+v/%v", verified, err)
	}
	if _, err := codec.RequireVerified(t.Context(), documentReader{content: content}, document.ID, nil); err == nil {
		t.Fatal("nil document verifier accepted")
	}
	if _, err := codec.Require(t.Context(), absentDocumentReader{}, document.ID); err == nil {
		t.Fatal("absent required document accepted")
	}
	resolved, found, err := codec.Resolve(t.Context(), codecAliasReader{
		documentReader: documentReader{content: content}, alias: document.ID,
	}, "current")
	if err != nil || !found || resolved.ID != document.ID {
		t.Fatalf("resolved document = %+v/%v/%v", resolved, found, err)
	}
	if _, found, err := codec.Resolve(t.Context(), codecAliasReader{}, "missing"); err != nil || found {
		t.Fatalf("absent alias = %v/%v", found, err)
	}
	if _, found, err := codec.Resolve(t.Context(), missingAliasTargetReader{codecAliasReader{alias: document.ID}}, "broken"); err == nil || !found {
		t.Fatalf("missing target = %v/%v", found, err)
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
