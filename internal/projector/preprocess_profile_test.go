package projector

import (
	"math"
	"strings"
	"testing"
)

func TestMediaPreprocessProfileCatalog(t *testing.T) {
	first, found, err := catalogMediaPreprocessProfile(qwen2VLProjectorType)
	if err != nil || !found {
		t.Fatalf("first profile = (%+v, %t, %v)", first, found, err)
	}
	second, found, err := catalogMediaPreprocessProfile(qwen3VLProjectorType)
	if err != nil || !found || second.ID != first.ID {
		t.Fatalf("second profile = (%+v, %t, %v), want %s", second, found, err, first.ID)
	}
	if _, err := first.Content(); err != nil {
		t.Fatal(err)
	}
	if _, found, err := catalogMediaPreprocessProfile("unbound"); err != nil || found {
		t.Fatalf("unbound profile = (%t, %v)", found, err)
	}
}

func TestAudioProcessorProfileIdentity(t *testing.T) {
	profile, found, err := catalogMediaPreprocessProfile(gemma4VisionTowerProjectorType)
	if err != nil || !found {
		t.Fatalf("tower profile: found=%t err=%v", found, err)
	}
	if _, err := NewAudioProjectionProfile(profile.AudioAttentionRopeFreqBase); err != nil {
		t.Fatal(err)
	}
	changed := profile
	changed.AudioAttentionRopeFreqBase *= 2
	if _, err := changed.Content(); err == nil {
		t.Fatal("changed audio policy retained the accepted identity")
	}
	for _, invalid := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		changed.AudioAttentionRopeFreqBase = invalid
		if _, err := mediaPreprocessProfileCodec.New(changed); err == nil {
			t.Fatalf("invalid audio-only policy %v accepted", invalid)
		}
	}
	legacy, found, err := catalogMediaPreprocessProfile(qwen2VLProjectorType)
	if err != nil || !found {
		t.Fatal("legacy raster profile missing")
	}
	content, err := legacy.Content()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(content.Data), "audio_attention") {
		t.Fatal("audio extension rewrote legacy raster profile bytes")
	}
}
