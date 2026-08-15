package diffusionimage

import (
	"testing"
)

func TestResidentSessionPolicyUsesOnlyCompiledGeometry(t *testing.T) {
	first, err := SessionPolicy(Request{Seed: 7, Steps: 2, Height: 64, Width: 64})
	if err != nil {
		t.Fatal(err)
	}
	second, err := SessionPolicy(Request{Seed: 99, Steps: 8, Height: 64, Width: 64})
	if err != nil {
		t.Fatal(err)
	}
	other, err := SessionPolicy(Request{Seed: 7, Steps: 2, Height: 32, Width: 64})
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first == other {
		t.Fatalf("resident policies first=%q second=%q other=%q", first, second, other)
	}
}
