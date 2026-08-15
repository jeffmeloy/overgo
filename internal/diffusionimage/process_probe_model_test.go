//go:build modeltest

package diffusionimage

import (
	"fmt"
	"testing"
)

func TestRealForwardProcessProbe(t *testing.T) {
	golden := loadGolden[struct {
		B, C, H, W int
		X, Out     []float32
	}](t, "real_forward")
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(golden.X, golden.B, golden.H, golden.W)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "real forward", got, golden.Out, tolReal)
	fmt.Println("SIMPLEDIFFUSION_PROCESS_PROBE exact=true")
}
