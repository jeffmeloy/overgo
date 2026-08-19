package protection

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
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
	store, err := repodb.Open(t.TempDir())
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
	value := testutil.ArtifactID(t, artifact.KindEvidence, "sealed golden")
	output := testutil.ArtifactID(t, artifact.KindEvidence, "external golden output")
	spec := SealedWriteSpec{
		Kind: SealedGolden, Name: "evaluation/reference-output", Value: value,
		Provenance: &ExternalProvenance{Tool: "reference-generator", Version: "1.0.0", Output: output},
	}
	candidate, err := SignSealedWrite(candidatePrivate, spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishSealed(context.Background(), server.Client(), server.URL, candidate); err == nil {
		t.Fatal("candidate principal wrote sealed authority")
	}
	if _, ok, err := store.ResolveAlias(context.Background(), "sealed/golden/evaluation/reference-output"); err != nil || ok {
		t.Fatalf("candidate write changed authority state: %v, %v", ok, err)
	}
	write, err := SignSealedWrite(private, spec)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := PublishSealed(context.Background(), server.Client(), server.URL, write)
	if err != nil || stored != write.ID {
		t.Fatalf("authority write = %s, %v; want %s", stored, err, write.ID)
	}
	if _, err := SignSealedWrite(private, SealedWriteSpec{Kind: SealedGolden, Name: "missing-provenance", Value: value}); err == nil {
		t.Fatal("golden without producing tool and version accepted")
	}
}
