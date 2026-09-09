package trainingdata

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestTextTransform(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var identities []artifact.ID
	for _, lowercase := range []bool{false, true} {
		transform, err := NewTextTransform(lowercase)
		if err != nil {
			t.Fatal(err)
		}
		identities = append(identities, transform.ID)
		content, err := transform.Content()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: transform.ID.String(), Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		reloaded, err := RequireTextTransform(t.Context(), store, transform.ID)
		if err != nil || reloaded != transform {
			t.Fatalf("reload: %v", err)
		}
		const raw = "ÉCOLE Straße\nAudio"
		want := raw
		if lowercase {
			want = "école straße\naudio"
		}
		got, err := reloaded.Apply(raw)
		if err != nil || got != want {
			t.Fatalf("apply: %q %v", got, err)
		}
		if _, err := reloaded.Apply("\xff"); err == nil {
			t.Fatal("malformed UTF-8 accepted")
		}
		reloaded.Lowercase = !reloaded.Lowercase
		if _, err := reloaded.Apply(raw); err == nil {
			t.Fatal("identity drift accepted")
		}
		reloaded = transform
		reloaded.Unicode = "different"
		if _, err := reloaded.Content(); err == nil {
			t.Fatal("Unicode version drift accepted")
		}
	}
	if identities[0] == identities[1] {
		t.Fatal("distinct transforms share an identity")
	}
	if _, err := (TextTransform{}).Apply("text"); err == nil {
		t.Fatal("zero transform accepted")
	}
}
