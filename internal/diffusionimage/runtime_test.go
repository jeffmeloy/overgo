package diffusionimage

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestResidentSessionKeyUsesOnlyCompiledGeometry(t *testing.T) {
	first := testcheck.Must((Request{Seed: 7, Steps: 2, Height: 64, Width: 64}).SessionKey())(t)
	second := testcheck.Must((Request{Seed: 99, Steps: 8, Height: 64, Width: 64}).SessionKey())(t)
	other := testcheck.Must((Request{Seed: 7, Steps: 2, Height: 32, Width: 64}).SessionKey())(t)
	if first != second || first == other {
		t.Fatalf("resident policies first=%q second=%q other=%q", first, second, other)
	}
}
