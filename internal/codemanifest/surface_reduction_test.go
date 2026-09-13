package codemanifest

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestSurfaceManifestRoundTrip(t *testing.T) {
	for _, large := range []bool{false, true} {
		t.Run(strconv.FormatBool(large), func(t *testing.T) {
			value := fixtureManifest()
			if large {
				boundary := Uncertainty{Kind: UncertaintyAnalysis, Path: "internal/fixture/0.go", Reason: strings.Repeat("x", maxPathBytes)}
				encoded, err := json.Marshal(boundary)
				if err != nil {
					t.Fatal(err)
				}
				value.Uncertainty = make([]Uncertainty, artifact.MaxContentBytes/len(encoded)+1)
				for index := range value.Uncertainty {
					value.Uncertainty[index] = boundary
					value.Uncertainty[index].Path = "internal/fixture/" + strconv.Itoa(index) + ".go"
				}
			}
			manifest, err := identifyManifest(value)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			if (len(encoded) > artifact.MaxContentBytes) != large {
				t.Fatalf("graph bytes=%d, storage limit=%d, large=%t", len(encoded), artifact.MaxContentBytes, large)
			}
			replayed, err := Parse(encoded)
			if err != nil || replayed.ID != manifest.ID {
				t.Fatalf("analysis round-trip ID=%s, want %s: %v", replayed.ID, manifest.ID, err)
			}
			if _, err := Parse(append(encoded, '\n')); err == nil {
				t.Fatal("noncanonical analysis bytes accepted")
			}
			if _, err := manifest.Content(); (err != nil) != large {
				t.Fatalf("storage admission large=%t: %v", large, err)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			if _, err := PublishDigest(t.Context(), store, manifest); err != nil {
				t.Fatal(err)
			}
			if _, err := PublishDigest(t.Context(), store, manifest); err != nil {
				t.Fatalf("repeat digest publication: %v", err)
			}
			digest, err := LoadDigest(t.Context(), store, manifest.ID)
			if err != nil || digest.ID != manifest.ID {
				t.Fatalf("digest round-trip: %+v, %v", digest, err)
			}
			if _, err := Load(t.Context(), store, manifest.ID); err == nil {
				t.Fatal("digest publication stored the full graph")
			}
			manifest.SourceIdentity = strings.Repeat("f", len(fixtureDigest))
			if err := manifest.Validate(); err == nil {
				t.Fatal("changed source retained its old identity")
			}
		})
	}
}
