package diffusionimage

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestResidentSessionKeyUsesOnlyCompiledGeometry(t *testing.T) {
	first, err := (Request{Seed: 7, Steps: 2, Height: 64, Width: 64}).SessionKey()
	testcheck.NoError(t, err)
	second, err := (Request{Seed: 99, Steps: 8, Height: 64, Width: 64}).SessionKey()
	testcheck.NoError(t, err)
	other, err := (Request{Seed: 7, Steps: 2, Height: 32, Width: 64}).SessionKey()
	testcheck.NoError(t, err)
	if first != second || first == other {
		t.Fatalf("resident policies first=%q second=%q other=%q", first, second, other)
	}
}
