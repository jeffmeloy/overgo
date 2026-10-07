//go:build modeltest

package main

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/composition"
	"overgo/internal/dataroot"
	"overgo/internal/rae"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/vitencoder"
)

// variedPicture draws a square picture, channel-first in [0, 1], of coloured
// blocks under a smooth ramp, different for each seed: enough variety that the
// tokens of a few of them span their space.
func variedPicture(side int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed))
	plane := side * side
	picture := make([]float32, 3*plane)
	const block = 16
	for by := 0; by < side; by += block {
		for bx := 0; bx < side; bx += block {
			colour := [3]float32{rng.Float32(), rng.Float32(), rng.Float32()}
			texture := rng.Float32()
			for y := by; y < min(by+block, side); y++ {
				for x := bx; x < min(bx+block, side); x++ {
					ramp := 0.25 * float32(x+y) / float32(2*side)
					stripe := texture * 0.2 * float32((x/2+y/3)%2)
					for channel, value := range colour {
						picture[channel*plane+y*side+x] = min(max(0.6*value+ramp+stripe, 0), 1)
					}
				}
			}
		}
	}
	return picture
}

// TestEncoderBridgeIsJudgedByHeldOutReconstruction holds the route's fit and
// its judge to a case with a known answer: the decoder's own encoder, read
// through its learned final norm, bridged to the same encoder's tokens
// without that norm's scale and shift. The two differ by a scale and a shift
// of each channel, which an affine bridge undoes, so the fit leaves almost
// nothing unexplained and, on a picture the fit never saw, the decoder draws
// from the bridged tokens what it draws from its own encoder's.
func TestEncoderBridgeIsJudgedByHeldOutReconstruction(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	collection := roots.ResolveModelPath("RAE-collections")
	resolved, held, err := rae.ResolveCollection(collection)
	if err != nil || !held {
		testskip.NotApplicable(t, "the RAE collection or its encoder is absent: "+collection)
	}
	if _, err := os.Stat(filepath.Join(resolved.EncoderDirectory(), "model.safetensors")); err != nil {
		testskip.NotApplicable(t, "the collection's encoder weights are absent: "+resolved.EncoderDirectory())
	}
	testskip.Short(t, "loads the collection's 1.7 GB decoder")
	pair, err := rae.LoadAutoencoder(collection)
	if err != nil {
		t.Fatal(err)
	}
	source, err := vitencoder.Load(resolved.EncoderDirectory())
	if err != nil {
		t.Fatal(err)
	}
	// Eight pictures of 256 tokens to fit a map 769 wide, and one other to judge on.
	const fitted = 8
	frames := make([]bridgeFrame, fitted+1)
	for index := range frames {
		if frames[index], err = bridgeTokens(t.Context(), pair, source, variedPicture(pair.Side(), uint64(index)+1)); err != nil {
			t.Fatal(err)
		}
	}
	_, _, sourceWidth := source.Shape()
	targetWidth, _ := pair.Latent()
	fit, err := fitEncoderBridge(frames[:fitted], sourceWidth, targetWidth)
	if err != nil {
		t.Fatal(err)
	}
	verdict, err := judgeEncoderBridge(t.Context(), pair, fit.Bridge, frames[fitted:])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("fit residual %.2e (%s); on the unseen picture own encoder %.2f dB, bridged %.2f dB, feature residual %.2e",
		fit.Residual, fit.Rung, verdict.OwnEncoderDB, verdict.BridgedDB, verdict.FeatureResidual)
	if fit.Residual > 1e-2 || fit.Rung == composition.AdapterBridge {
		t.Fatalf("an affine relation was fitted with residual %.3g, rung %s", fit.Residual, fit.Rung)
	}
	if verdict.Frames != 1 || verdict.FeatureResidual > 5e-2 || math.Abs(verdict.DifferenceDB) > 0.5 {
		t.Fatalf("the bridged encoder is judged %+v, want what the decoder's own encoder gives", verdict)
	}
	// The pair the bridge makes, as it is served, gives back what the decoder's own pair does.
	bridged, err := pair.Bridged(source, fit.Bridge)
	if err != nil {
		t.Fatal(err)
	}
	unseen := frames[fitted].picture
	own, err := pair.RoundTrip(t.Context(), unseen)
	if err != nil {
		t.Fatal(err)
	}
	through, err := bridged.RoundTrip(t.Context(), unseen)
	if err != nil {
		t.Fatal(err)
	}
	var worst float64
	for index, value := range own {
		worst = max(worst, math.Abs(float64(value)-float64(through[index])))
	}
	if worst > 2e-2 {
		t.Fatalf("the bridged pair's picture differs from the decoder's own pair's by up to %.3g of the range", worst)
	}
	// A bridge of another width does not fit the decoder, and too few pictures do not determine the map.
	if _, err := pair.Bridged(source, rae.EncoderBridge{Weight: make([]float32, 4), Bias: make([]float32, 2)}); err == nil {
		t.Fatal("a bridge of the wrong widths was joined to the decoder")
	}
	if _, err := fitEncoderBridge(frames[:2], sourceWidth, targetWidth); err == nil {
		t.Fatal("a map 769 wide was fitted from 512 tokens")
	}
}
