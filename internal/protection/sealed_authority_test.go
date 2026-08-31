package protection

import (
	"crypto/ed25519"
	"crypto/rand"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestCandidatePrincipalCannotWriteSealedAuthority(t *testing.T) {
	const requestBytes = 4096
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	_, candidatePrivate, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authority, err := NewSealedAuthority(public, store, SealedAuthorityConfig{MaxRequestBytes: requestBytes})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(authority)
	defer server.Close()
	fixtures := []SealedWriteSpec{
		{Kind: SealedGolden, Name: "evaluation/reference-output",
			Value: testutil.ArtifactID(t, artifact.KindEvidence, "sealed golden"),
			Provenance: &ExternalProvenance{Tool: "reference-generator", Version: "1.0.0",
				Output: testutil.ArtifactID(t, artifact.KindEvidence, "external golden output")}},
		{Kind: SealedEvaluator, Name: "evaluation/exact",
			Value: testutil.ArtifactID(t, artifact.KindEvidence, "sealed evaluator")},
		{Kind: SealedPromotionPolicy, Name: "promotion/default",
			Value: testutil.ArtifactID(t, artifact.KindProfile, "sealed promotion policy")},
		{Kind: SealedChampionAlias, Name: "inference/default",
			Value: testutil.ArtifactID(t, artifact.KindModel, "sealed champion")},
	}
	for _, spec := range fixtures {
		candidate, err := SignSealedWrite(candidatePrivate, spec)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := PublishSealed(t.Context(), server.Client(), server.URL, candidate); err == nil {
			t.Fatalf("candidate principal wrote %s authority", spec.Kind)
		}
		alias := "sealed/" + string(spec.Kind) + "/" + spec.Name
		if _, ok, err := store.ResolveAlias(t.Context(), alias); err != nil || ok {
			t.Fatalf("candidate %s write changed authority state: %v, %v", spec.Kind, ok, err)
		}
		write, err := SignSealedWrite(private, spec)
		if err != nil {
			t.Fatal(err)
		}
		stored, err := PublishSealed(t.Context(), server.Client(), server.URL, write)
		if err != nil || stored != write.ID {
			t.Fatalf("%s authority write = %s, %v; want %s", spec.Kind, stored, err, write.ID)
		}
		if resolved, ok, err := store.ResolveAlias(t.Context(), alias); err != nil || !ok || resolved != write.ID {
			t.Fatalf("%s alias = %s, %v, %v; want %s", spec.Kind, resolved, ok, err, write.ID)
		}
	}
	if _, err := SignSealedWrite(private, SealedWriteSpec{
		Kind: SealedGolden, Name: "missing-provenance", Value: fixtures[0].Value,
	}); err == nil {
		t.Fatal("golden without producing tool and version accepted")
	}
	previous, err := SignSealedWrite(private, fixtures[0])
	if err != nil {
		t.Fatal(err)
	}
	updated := fixtures[0]
	updated.Previous = &previous.ID
	updated.Value = testutil.ArtifactID(t, artifact.KindEvidence, "updated sealed golden")
	updated.Provenance = &ExternalProvenance{Tool: "reference-generator", Version: "1.1.0",
		Output: testutil.ArtifactID(t, artifact.KindEvidence, "updated external golden output")}
	write, err := SignSealedWrite(private, updated)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishSealed(t.Context(), server.Client(), server.URL, write); err != nil {
		t.Fatal(err)
	}
	stale := updated
	stale.Value = testutil.ArtifactID(t, artifact.KindEvidence, "stale sealed golden")
	stale.Provenance = &ExternalProvenance{Tool: "reference-generator", Version: "1.2.0",
		Output: testutil.ArtifactID(t, artifact.KindEvidence, "stale external golden output")}
	staleWrite, err := SignSealedWrite(private, stale)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishSealed(t.Context(), server.Client(), server.URL, staleWrite); err == nil {
		t.Fatal("stale golden update changed sealed authority")
	}
}
