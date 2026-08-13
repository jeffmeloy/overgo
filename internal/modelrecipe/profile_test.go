package modelrecipe

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

const profileCatalogSemanticDigest = "308bb8455fb12dfb47666fbe347a149dd4970abf3ec3b74edac680c9073d9b92"
const architectureProfileFactCount = 130
const architectureProfileFactDigest = "38f3c3ef55c464a0ee365ba499189392bcb30949fa82890ffcc9e2cc233db08c"
const architectureProfileFactSchemaDigest = "8694762753f81f536904e5aba8e56458df94a2684111581e637aee2d8054a466"

const unsupportedProfilePolicyValue = ^uint8(0)

func seedProfileDocuments(t *testing.T) []ProfileDocument {
	t.Helper()
	names := model.SupportedArchitectures()
	documents := make([]ProfileDocument, 0, len(names))
	for _, name := range names {
		profile, ok := model.LookupArchitecture(name)
		if !ok {
			t.Fatalf("registered profile %q is absent", name)
		}
		document, err := NewProfileDocument(profile)
		if err != nil {
			t.Fatal(err)
		}
		documents = append(documents, document)
	}
	return documents
}

func TestRegisteredProfileDocumentsRoundTrip(t *testing.T) {
	documents := seedProfileDocuments(t)
	if len(documents) != len(model.SupportedArchitectures()) {
		t.Fatalf("documents = %d", len(documents))
	}
	for _, document := range documents {
		content, err := document.Content()
		if err != nil {
			t.Fatalf("%s: %v", document.Architecture, err)
		}
		parsed, err := ParseProfileDocument(content)
		if err != nil {
			t.Fatalf("%s: %v", document.Architecture, err)
		}
		if parsed.ID != document.ID || parsed.Policy != document.Policy ||
			!reflect.DeepEqual(parsed.Provenance, document.Provenance) {
			t.Fatalf("%s: profile round trip drifted", document.Architecture)
		}
	}
}

func TestProfileProvenanceHasExactTypedCoverage(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	base, err := CatalogProfileProvenance(profile)
	if err != nil {
		t.Fatal(err)
	}
	facts := slices.Clone(architectureProfileFacts)
	if len(base) != len(facts) || len(base) != architectureProfileFactCount {
		t.Fatalf("profile facts = %d", len(base))
	}
	digest := sha256.New()
	for _, fact := range facts {
		fmt.Fprintln(digest, fact)
	}
	if got := fmt.Sprintf("%x", digest.Sum(nil)); got != architectureProfileFactDigest {
		t.Fatalf("profile fact digest = %s", got)
	}
	schema, err := json.Marshal(architectureProfileFactSpecs)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(schema)); got != architectureProfileFactSchemaDigest {
		t.Fatalf("profile fact schema digest = %s", got)
	}
	first, err := NewProfileDocumentWithProvenance(profile, base)
	if err != nil {
		t.Fatal(err)
	}
	changed := slices.Clone(base)
	changed[0].Origin = ProfileFactExternal
	changed[0].SourceField = "parity.fixture"
	second, err := NewProfileDocumentWithProvenance(profile, changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID == second.ID {
		t.Fatal("provenance-only change did not affect profile identity")
	}
	if _, err := NewProfileDocumentWithProvenance(profile, base[:len(base)-1]); err == nil {
		t.Fatal("incomplete profile provenance accepted")
	}
	invalid := slices.Clone(base)
	invalid[0].SourceField = ""
	if _, err := NewProfileDocumentWithProvenance(profile, invalid); err == nil {
		t.Fatal("profile provenance without source accepted")
	}
}

func TestProfileDocumentRejectsInvalidPolicy(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	profile.Forward = model.ForwardProgram{Operation: model.ForwardOperation(unsupportedProfilePolicyValue)}
	if _, err := NewProfileDocument(profile); err == nil {
		t.Fatal("invalid profile document accepted")
	}
	content, err := json.Marshal(profileBody{
		Version: ProfileVersion, Architecture: profile.Name, Policy: profile,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseProfileDocument(content); err == nil {
		t.Fatal("invalid serialized profile accepted")
	}
}

func TestRegisteredProfileSemanticDigest(t *testing.T) {
	documents := seedProfileDocuments(t)
	digest := sha256.New()
	for _, document := range documents {
		fmt.Fprintln(digest, document.ID)
	}
	if got := fmt.Sprintf("%x", digest.Sum(nil)); got != profileCatalogSemanticDigest {
		t.Fatalf("profile catalog digest = %s", got)
	}
}

func TestLoadProfileRequiresStoredDerivationLineage(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	content, err := profileContent(document)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/profile/unbound",
		[]artifact.Content{content},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := loadProfile(ctx, store, document.ID); err == nil {
		t.Fatal("profile without derivation lineage loaded")
	}
}

func TestProfilePublicationRequiresExternalDerivation(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	profile, _ := model.LookupArchitecture("llama")
	provenance, err := CatalogProfileProvenance(profile)
	if err != nil {
		t.Fatal(err)
	}
	externalID := testutil.ArtifactID(t, artifact.KindEvidence, "external-derivation")
	for index := range provenance {
		provenance[index].Origin = ProfileFactExternal
		provenance[index].SourceField = "external.profile"
		provenance[index].DerivationID = externalID
	}
	document, err := NewProfileDocumentWithProvenance(profile, provenance)
	if err != nil {
		t.Fatal(err)
	}
	contents, lineage, err := profilePublicationFacts(document)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("fixture/catalog/external", contents, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err == nil {
		t.Fatal("profile with absent external derivation committed")
	}
}

func TestProfileDocumentRejectsNonCanonicalContent(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	content = append(content[:len(content)-1], []byte(",\"extra\":true}")...)
	if _, err := ParseProfileDocument(content); err == nil {
		t.Fatal("unknown profile field accepted")
	}
}

func commitFixtureModel(
	t *testing.T,
	ctx context.Context,
	store artifact.Repository,
	key string,
	payload string,
) artifact.ID {
	t.Helper()
	data := []byte(payload)
	id := testutil.ArtifactBytesID(t, artifact.KindModel, data)
	if _, err := store.Commit(ctx, artifact.Batch{Key: key, Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data,
	}}}); err != nil {
		t.Fatal(err)
	}
	return id
}
