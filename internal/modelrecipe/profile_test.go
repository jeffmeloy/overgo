package modelrecipe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

const profileCatalogSemanticDigest = "201721bec53a80ab0499c4a43c0fd1ad0489ea804186f8a368492153ebf71af7"
const architectureProfileFactCount = 128
const architectureProfileFactDigest = "4ac84dcd0df87bb4a5d22be0d2fa37c4935741f9d732ad9becfbeb909a925d0f"
const architectureProfileFactSchemaDigest = "f8b015a77ae2ffb753458dd51d70c3aaf9b4b678b499e95e1798ba86fd9c89d7"

const unsupportedProfilePolicyValue = ^uint8(0)

func TestRegisteredProfileDocumentsRoundTrip(t *testing.T) {
	documents, err := SeedProfileDocuments()
	if err != nil {
		t.Fatal(err)
	}
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
	facts := ArchitectureProfileFacts()
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
	schema, err := json.Marshal(ArchitectureProfileFactSchema())
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

func TestLegacyProfileDocumentRemainsReadable(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	content, err := json.Marshal(profileBody{
		Version: LegacyProfileVersion, Architecture: profile.Name, Policy: profile,
	})
	if err != nil {
		t.Fatal(err)
	}
	document, err := ParseProfileDocument(content)
	if err != nil {
		t.Fatal(err)
	}
	if document.Version != LegacyProfileVersion || len(document.Provenance) != 0 {
		t.Fatalf("legacy profile = %+v", document)
	}
	encoded, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, content) {
		t.Fatal("legacy profile encoding changed")
	}
}

func TestProfileDocumentRejectsInvalidPolicy(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	profile.Forward = model.ForwardPolicy(unsupportedProfilePolicyValue)
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
	documents, err := SeedProfileDocuments()
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	for _, document := range documents {
		fmt.Fprintln(digest, document.ID)
	}
	if got := fmt.Sprintf("%x", digest.Sum(nil)); got != profileCatalogSemanticDigest {
		t.Fatalf("profile catalog digest = %s", got)
	}
}

func TestPublishProfilesSeedsRegisteredAliases(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_, documents, err := PublishProfiles(ctx, store, "fixture/profiles")
	if err != nil {
		t.Fatal(err)
	}
	document, ok, err := RegisteredProfile(ctx, store, "llama")
	if err != nil || !ok || document.Architecture != "llama" {
		t.Fatalf("registered profile = (%s, %v, %v)", document.Architecture, ok, err)
	}
	if len(documents) != len(model.SupportedArchitectures()) {
		t.Fatalf("documents = %d", len(documents))
	}
	derivationID := document.Provenance[0].DerivationID
	content, ok, err := store.Content(ctx, derivationID)
	if err != nil || !ok || content.Descriptor.MediaType != CatalogProfileDerivationMediaType {
		t.Fatalf("profile derivation content = (%+v, %t, %v)", content.Descriptor, ok, err)
	}
	derivation, err := ParseCatalogProfileDerivation(content.Data)
	if err != nil || derivation.ID != derivationID || derivation.Architecture != "llama" {
		t.Fatalf("profile derivation = (%+v, %v)", derivation, err)
	}
	parents, err := store.Parents(ctx, document.ID)
	if err != nil || len(parents) != 1 || parents[0].Parent != derivationID {
		t.Fatalf("profile derivation lineage = (%+v, %v)", parents, err)
	}
}

func TestRegisteredProfileRequiresStoredDerivationLineage(t *testing.T) {
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
	content, err := ProfileContent(document)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/profile/unbound",
		[]artifact.Content{content},
		nil,
		[]artifact.AliasBinding{{Name: registeredProfileAlias("llama"), Target: document.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := RegisteredProfile(ctx, store, "llama"); err == nil {
		t.Fatal("profile without derivation lineage loaded")
	}
}

func TestPublishProfileCatalogReplacesBootstrapAuthority(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	bootstrap, _ := model.LookupArchitecture("llama")
	bootstrapDocument, err := NewProfileDocument(bootstrap)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishProfileCatalog(ctx, store, "fixture/catalog/bootstrap", []ProfileDocument{bootstrapDocument}); err != nil {
		t.Fatal(err)
	}
	candidate := bootstrap
	candidate.DeciSparse = true
	candidateDocument, err := NewProfileDocument(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishProfileCatalog(ctx, store, "fixture/catalog/candidate", []ProfileDocument{candidateDocument}); err != nil {
		t.Fatal(err)
	}
	stored, ok, err := RegisteredProfile(ctx, store, "llama")
	if err != nil || !ok || stored.ID != candidateDocument.ID || !stored.Policy.DeciSparse {
		t.Fatalf("stored profile = (%+v, %v, %v)", stored, ok, err)
	}
	registered, _ := model.LookupArchitecture("llama")
	if registered != bootstrap {
		t.Fatal("RepoDB profile publication mutated bootstrap fallback")
	}
}

func TestPublishProfileCatalogRejectsDuplicateArchitecture(t *testing.T) {
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
	if _, err := PublishProfileCatalog(
		ctx, store, "fixture/catalog/duplicate", []ProfileDocument{document, document},
	); err == nil {
		t.Fatal("duplicate profile architecture accepted")
	}
}

func TestPublishProfileCatalogRequiresExternalDerivation(t *testing.T) {
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
	if _, err := PublishProfileCatalog(ctx, store, "fixture/catalog/external", []ProfileDocument{document}); err == nil {
		t.Fatal("profile with absent external derivation committed")
	}
}

func TestProfileCandidateParityPromotion(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := commitFixtureModel(t, ctx, store, "fixture/profile/model", "parity-model")
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := inferenceWithProfileFixture(
		modelID, document.ID, recipe.PlacementHost, DecodeSessionCapacity,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishProfileCandidate(
		ctx, store, "fixture/profile/candidate", definition, document,
	); err != nil {
		t.Fatal(err)
	}
	if _, bound, err := store.ResolveAlias(ctx, legacyRecipeProfileAlias(definition.ID)); err != nil || bound {
		t.Fatalf("legacy profile alias = (%v, %v)", bound, err)
	}
	parents, err := store.Parents(ctx, definition.ID)
	if err != nil || len(parents) != len(definition.Dependencies) {
		t.Fatalf("dependency lineage = (%+v, %v)", parents, err)
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}
	_, _, evidence, err := ValidateProfileCandidate(
		ctx, store, "fixture/profile/validated", definition, spec, model.Weights{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Profile != document.ID || evidence.Recipe != definition.ID {
		t.Fatalf("evidence = %+v", evidence)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/profile/verification")
	if _, _, err := ActivateProfileCandidate(
		ctx, store, "fixture/profile/active", definition, verification, nil,
	); err != nil {
		t.Fatal(err)
	}
	plan, ok, err := compileActiveFixture(ctx, store, modelID, spec, model.Weights{})
	if err != nil || !ok || plan.Model.Profile() != profile {
		t.Fatalf("active profile plan = (%+v, %v, %v)", plan.Model, ok, err)
	}
	activeProfile, ok, err := ActiveProfile(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || activeProfile.ID != document.ID {
		t.Fatalf("active profile = (%s, %v, %v)", activeProfile.ID, ok, err)
	}
}

func TestProfileCandidateRejectsParityDrift(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := commitFixtureModel(t, ctx, store, "fixture/drift/model", "drift-model")
	profile, _ := model.LookupArchitecture("llama")
	profile.Attention = model.AttentionLFM2
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := inferenceWithProfileFixture(
		modelID, document.ID, recipe.PlacementHost, DecodeSessionCapacity,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishProfileCandidate(
		ctx, store, "fixture/drift/candidate", definition, document,
	); err != nil {
		t.Fatal(err)
	}
	_, _, _, err = ValidateProfileCandidate(
		ctx, store, "fixture/drift/validated", definition,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}, model.Weights{},
	)
	if err == nil {
		t.Fatal("drifted profile passed parity validation")
	}
}

func TestCompileWithProfileMatchesRegistry(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "profile-model")
	profile, ok := model.LookupArchitecture("llama")
	if !ok {
		t.Fatal("llama profile is absent")
	}
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := inferenceWithProfileFixture(
		modelID, document.ID, recipe.PlacementHost, DecodeSessionCapacity,
	)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := VerifyProfileParity(
		definition, document,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}, model.Weights{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Model.Profile().Name != "llama" || plan.Model.LayerCount() != 1 {
		t.Fatalf("plan = %+v", plan.Model)
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
