package closureledger

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestDocumentRoundTripPreservesExactBinding(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner")
	fixture := testutil.ArtifactID(t, artifact.KindEvidence, "fixture")
	binding := testSourceBinding(owner)
	document, err := New(
		"decode.page_size", json.RawMessage(`{ "tokens": 256, "class": "decode" }`),
		TierImplementation, StatusClosed, "Fixed cache-page implementation contract.", []SourceBinding{binding},
		"Runtime refuses cache pages outside the compiled capacity class.",
		"Re-evaluate when the cache ABI supports variable page geometry.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(document.Value) != `{"class":"decode","tokens":256}` || document.Bindings[0] != binding {
		t.Fatalf("canonical value = %s", document.Value)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(content.Data)
	if err != nil || parsed.ID != document.ID {
		t.Fatalf("round trip = (%+v, %v)", parsed, err)
	}
}

func TestDocumentRejectsIncompleteClosureEvidence(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner")
	fixture := testutil.ArtifactID(t, artifact.KindEvidence, "fixture")
	binding := testSourceBinding(owner)
	newDocument := func(value json.RawMessage, understanding string, bindings []SourceBinding, fixtureID artifact.ID) error {
		_, err := New(
			"decode.page_size", value, TierDerivationBlocked, StatusOpen, understanding, bindings,
			"Derive from measured cache boundaries.", "Re-evaluate after cache ABI changes.", fixtureID,
		)
		return err
	}
	if err := newDocument(nil, "Open cache geometry.", []SourceBinding{binding}, fixture); err == nil {
		t.Fatal("missing value accepted")
	}
	if err := newDocument(json.RawMessage(`256`), "", []SourceBinding{binding}, fixture); err == nil {
		t.Fatal("missing understanding accepted")
	}
	if err := newDocument(json.RawMessage(`256`), "Open cache geometry.", nil, fixture); err == nil {
		t.Fatal("missing binding accepted")
	}
	invalid := binding
	invalid.SourceID = "invalid"
	if err := newDocument(json.RawMessage(`256`), "Open cache geometry.", []SourceBinding{invalid}, fixture); err == nil {
		t.Fatal("invalid source accepted")
	}
	if err := newDocument(json.RawMessage(`256`), "Open cache geometry.", []SourceBinding{binding}, artifact.ID{}); err == nil {
		t.Fatal("missing fixture accepted")
	}
	if err := newDocument(json.RawMessage(`256`), "Open cache geometry.", []SourceBinding{binding, binding}, fixture); err == nil {
		t.Fatal("duplicate binding accepted")
	}
	if _, err := New(
		"decode.page_size", json.RawMessage(`256`), TierMathematicalFact, StatusOpen,
		"Serialized format fact.", []SourceBinding{binding}, "Defined by the serialized format.",
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
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner")
	fixture := testutil.ArtifactID(t, artifact.KindEvidence, "fixture")
	document, err := New(
		"decode.page_size", json.RawMessage(`256`), TierImplementation, StatusClosed,
		"Fixed cache-page implementation contract.", []SourceBinding{testSourceBinding(owner)},
		"Fixed-capacity append validates page geometry.",
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
	loaded, ok, err := documentCodec.Read(ctx, store, document.ID)
	if err != nil || !ok || loaded.ID != document.ID {
		t.Fatalf("loaded document = (%+v, %t, %v)", loaded, ok, err)
	}
}

func testSourceBinding(owner artifact.ID) SourceBinding {
	digest := sha256.Sum256([]byte("internal/cache/page.go"))
	return SourceBinding{
		Kind: BindingConstant, Package: "internal/cache", File: "internal/cache/page.go",
		Scope: "package", Name: "pageSize", Line: 12, Expression: "1 << 8",
		SourceID: hex.EncodeToString(digest[:]), Owner: owner,
	}
}
