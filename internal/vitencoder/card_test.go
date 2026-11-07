package vitencoder

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
)

// TestTheCardEncoderGivesTheHostTokens holds the encoder on the card to the encoder on the host for one picture: every
// token webssl-dino300m gives within float32 accumulation order of the host's, each token pointing its way. It logs
// one frame's time on each and the card memory the process peaked at.
func TestTheCardEncoderGivesTheHostTokens(t *testing.T) {
	// Serial: holds the card.
	cudatest.Require(t)
	const absolute, alignment = 2e-3, 0.999999
	picture := image.NewRGBA(image.Rect(0, 0, 320, 256))
	for y := range 256 {
		for x := range 320 {
			picture.Set(x, y, color.RGBA{R: uint8((x*7 + y*13 + (x*y)%17) % 256), G: uint8((x*3 + y*29) % 256), B: uint8((x ^ y) % 256), A: 255})
		}
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, picture))
	encoder := testcheck.Must(Load(modeltest.Directory(t, "webssl-dino300m-full2b-224")))(t)
	started := time.Now()
	want := testcheck.Must(encoder.Tokens(encoded.Bytes()))(t)
	host := time.Since(started)
	card := testcheck.Must(encoder.OpenCard(t.Context(), device.DefaultOrdinal()))(t)
	// The first frame compiles the card's launches; the second is the one a composite's frames take.
	testcheck.Must(card.Tokens(t.Context(), encoded.Bytes()))(t)
	started = time.Now()
	got := testcheck.Must(card.Tokens(t.Context(), encoded.Bytes()))(t)
	onCard := time.Since(started)
	testcheck.NoError(t, card.Close())
	_, _, hidden := encoder.Shape()
	if len(got) != len(want) {
		t.Fatalf("the card gives %d values, the host %d", len(got), len(want))
	}
	worst, least := 0.0, 1.0
	for token := range len(want) / hidden {
		var dot, gotNorm, wantNorm float64
		for channel := token * hidden; channel < (token+1)*hidden; channel++ {
			g, w := float64(got[channel]), float64(want[channel])
			worst = max(worst, math.Abs(g-w))
			dot, gotNorm, wantNorm = dot+g*w, gotNorm+g*g, wantNorm+w*w
		}
		least = min(least, dot/math.Sqrt(gotNorm*wantNorm))
	}
	t.Logf("%d tokens on the card stand within %.3g of the host's, cosine at least %.9f; a frame takes %s on the card, %s on the host; the process peaked at %d card bytes",
		len(want)/hidden, worst, least, onCard.Round(time.Millisecond), host.Round(time.Millisecond), driver.ProcessPeakBytes())
	if worst > absolute || least < alignment {
		t.Errorf("the card's tokens stand %.3g from the host's, cosine %.9f", worst, least)
	}
}
