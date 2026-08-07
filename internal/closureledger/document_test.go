package closureledger

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestDocumentRoundTripAndCanonicalValue(t *testing.T) {
	owner := fixtureID(t, artifact.KindFile, "owner")
	fixture := fixtureID(t, artifact.KindEvidence, "fixture")
	document, err := New(
		"decode.page_size", json.RawMessage(`{ "tokens": 256, "class": "decode" }`),
		TierImplementation, StatusClosed, []artifact.ID{owner},
		"Runtime refuses cache pages outside the compiled capacity class.",
		"Re-evaluate when the cache ABI supports variable page geometry.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(document.Value) != `{"class":"decode","tokens":256}` {
		t.Fatalf("canonical value = %s", document.Value)
	}
	content, err := document.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(content)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed, err)
	}
}

func TestDocumentRejectsMissingTypedFacts(t *testing.T) {
	owner := fixtureID(t, artifact.KindFile, "owner")
	fixture := fixtureID(t, artifact.KindEvidence, "fixture")
	newDocument := func(value json.RawMessage, owners []artifact.ID, fixtureID artifact.ID) error {
		_, err := New(
			"decode.page_size", value, TierDerivationBlocked, StatusOpen, owners,
			"Derive from measured cache boundaries.", "Re-evaluate after cache ABI changes.", fixtureID,
		)
		return err
	}
	if err := newDocument(nil, []artifact.ID{owner}, fixture); err == nil {
		t.Fatal("missing value accepted")
	}
	if err := newDocument(json.RawMessage(`256`), nil, fixture); err == nil {
		t.Fatal("missing owner accepted")
	}
	if err := newDocument(json.RawMessage(`256`), []artifact.ID{owner}, artifact.ID{}); err == nil {
		t.Fatal("missing fixture accepted")
	}
	if err := newDocument(json.RawMessage(`256`), []artifact.ID{owner, owner}, fixture); err == nil {
		t.Fatal("duplicate owner accepted")
	}
	if _, err := New(
		"decode.page_size", json.RawMessage(`256`), TierMathematicalFact, StatusOpen,
		[]artifact.ID{owner}, "Defined by the serialized format.",
		"Re-evaluate only with a new format version.", fixture,
	); err == nil {
		t.Fatal("tier/status contradiction accepted")
	}
}

func TestPublicationRequiresStoredOwnerAndFixture(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	owner := fixtureID(t, artifact.KindFile, "owner")
	fixture := fixtureID(t, artifact.KindEvidence, "fixture")
	document, err := New(
		"decode.page_size", json.RawMessage(`256`), TierImplementation, StatusClosed,
		[]artifact.ID{owner}, "Fixed-capacity append validates page geometry.",
		"Re-evaluate when page geometry changes.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := document.Batch("fixture/magic", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err == nil {
		t.Fatal("document with missing owner and fixture committed")
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/dependencies",
		Artifacts: []artifact.Descriptor{{ID: owner}, {ID: fixture}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	loaded, ok, err := Load(ctx, store, document.ID)
	if err != nil || !ok || loaded.ID != document.ID {
		t.Fatalf("loaded document = (%+v, %t, %v)", loaded, ok, err)
	}
}

func fixtureID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
