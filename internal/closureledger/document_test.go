package closureledger

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
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

func TestPermanentMagicGateRejectsIncompleteClosureEvidence(t *testing.T) {
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
	store, err := overgodb.Open(t.TempDir())
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

func TestActiveBindingsResolveExactOwnerAndValue(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner")
	fixture := testutil.ArtifactID(t, artifact.KindEvidence, "fixture")
	binding := testSourceBinding(owner)
	other := binding
	other.File = "internal/cache/other.go"
	alias, err := ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	otherAlias, err := ActiveAlias(other)
	if err != nil || alias == otherAlias {
		t.Fatalf("source aliases = (%q, %q, %v)", alias, otherAlias, err)
	}
	document := publishActiveDocument(t, ctx, store, binding, fixture, json.RawMessage(`256`), nil)

	loaded, found, err := ResolveActiveBinding(ctx, store, binding, json.RawMessage(` 256 `))
	if err != nil || !found || loaded.ID != document.ID {
		t.Fatalf("active binding = (%s, %t, %v), want %s", loaded.ID, found, err, document.ID)
	}
}

func TestActiveBindingsRejectAmbiguousOrStaleRows(t *testing.T) {
	t.Run("stale source", func(t *testing.T) {
		ctx, store, binding, fixture := activeFixture(t)
		publishActiveDocument(t, ctx, store, binding, fixture, json.RawMessage(`256`), nil)
		digest := sha256.Sum256([]byte("stale source"))
		binding.SourceID = hex.EncodeToString(digest[:])
		if _, found, err := ResolveActiveBinding(ctx, store, binding, json.RawMessage(`256`)); err == nil || found {
			t.Fatalf("stale source = (%t, %v)", found, err)
		}
	})
	t.Run("stale value", func(t *testing.T) {
		ctx, store, binding, fixture := activeFixture(t)
		publishActiveDocument(t, ctx, store, binding, fixture, json.RawMessage(`256`), nil)
		if _, found, err := ResolveActiveBinding(ctx, store, binding, json.RawMessage(`512`)); err == nil || found {
			t.Fatalf("stale value = (%t, %v)", found, err)
		}
	})
	t.Run("ambiguous declaration", func(t *testing.T) {
		ctx, store, binding, fixture := activeFixture(t)
		other := binding
		digest := sha256.Sum256([]byte("revised source"))
		other.SourceID = hex.EncodeToString(digest[:])
		other.Owner = testutil.ArtifactID(t, artifact.KindFile, "other owner")
		publishActiveDocument(t, ctx, store, binding, fixture, json.RawMessage(`256`), &other)
		if _, found, err := ResolveActiveBinding(ctx, store, binding, json.RawMessage(`256`)); err == nil || found {
			t.Fatalf("ambiguous declaration = (%t, %v)", found, err)
		}
	})
}

func activeFixture(t *testing.T) (context.Context, *overgodb.Store, SourceBinding, artifact.ID) {
	t.Helper()
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	owner := testutil.ArtifactID(t, artifact.KindFile, "owner")
	return ctx, store, testSourceBinding(owner), testutil.ArtifactID(t, artifact.KindEvidence, "fixture")
}

func publishActiveDocument(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	binding SourceBinding,
	fixture artifact.ID,
	value json.RawMessage,
	extra *SourceBinding,
) Document {
	t.Helper()
	bindings := []SourceBinding{binding}
	artifacts := []artifact.Descriptor{{ID: binding.Owner}, {ID: fixture}}
	if extra != nil {
		bindings = append(bindings, *extra)
		artifacts = append(artifacts, artifact.Descriptor{ID: extra.Owner})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "active/dependencies", Artifacts: artifacts}); err != nil {
		t.Fatal(err)
	}
	document, err := New(
		"decode.page_size", value, TierImplementation, StatusClosed,
		"Fixed cache-page implementation contract.", bindings,
		"Fixed-capacity append validates page geometry.",
		"Re-evaluate when page geometry changes.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	alias, err := ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := document.Batch("active/document", &artifact.AliasBinding{Name: alias, Target: document.ID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	return document
}

func testSourceBinding(owner artifact.ID) SourceBinding {
	digest := sha256.Sum256([]byte("internal/cache/page.go"))
	callsites := sha256.Sum256([]byte("internal/cache/use.go"))
	structure := sha256.Sum256([]byte("pageSize declaration"))
	return SourceBinding{
		Kind: BindingConstant, Package: "internal/cache", File: "internal/cache/page.go",
		Scope: "package", Name: "pageSize", Line: 12, Expression: "1 << 8",
		StructuralID: hex.EncodeToString(structure[:]), SourceID: hex.EncodeToString(digest[:]),
		CallsiteID: hex.EncodeToString(callsites[:]), Owner: owner,
	}
}

func TestStructuralAliasIgnoresLineAndOwner(t *testing.T) {
	binding := testSourceBinding(testutil.ArtifactID(t, artifact.KindFile, "first owner"))
	moved := binding
	moved.Line += 10
	moved.Owner = testutil.ArtifactID(t, artifact.KindFile, "second owner")
	left, err := ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	right, err := ActiveAlias(moved)
	if err != nil || left != right {
		t.Fatalf("structural aliases = (%s, %s, %v)", left, right, err)
	}
}
