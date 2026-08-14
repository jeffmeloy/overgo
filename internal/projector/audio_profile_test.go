package projector

import "testing"

func TestAudioProjectionProfileRoundTrip(t *testing.T) {
	profile, err := NewAudioProjectionProfile(10000)
	if err != nil {
		t.Fatal(err)
	}
	content, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseAudioProjectionProfile(content.Data)
	if err != nil || parsed != profile {
		t.Fatalf("audio projection profile = (%+v, %v)", parsed, err)
	}
	if _, err := NewAudioProjectionProfile(0); err == nil {
		t.Fatal("zero audio RoPE base accepted")
	}
}
