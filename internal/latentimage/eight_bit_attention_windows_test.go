//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestImageEightBitAttentionPresetDrawsFasterAtAPersonsSize holds the eight-bit-attention preset of Krea-2-Turbo and
// MOSS on the card at the size a person draws, Krea's 2048 x 2048 and MOSS's 1024 x 1024 (eight-bit-attention/
// images-wiring): a session given the preset by Reset computes its denoiser's attention in 8 bits, draws a picture
// other than its full-precision one, and draws it faster, each arm timed on its second picture, after the first's
// one-time work. Whether its pictures are as good is its suite's to read (eight-bit-attention/images).
func TestImageEightBitAttentionPresetDrawsFasterAtAPersonsSize(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	for _, model := range []struct {
		directory string
		side      int
	}{{"Krea-2-Turbo", 2048}, {"MOSS-Image-SFT-797fa8e678b4f", 1024}} {
		directory := testutil.ModelArtifactDir(t, model.directory)
		profile := testcheck.Must(ResolveProfile(directory))(t)
		request := Request{Prompt: "a red fox licking a vanilla ice cream cone in snow", Width: model.side, Height: model.side, Seed: 42}
		generator := testcheck.Must(LoadGenerator(ctx, directory, profile, request))(t)
		var pictures [2][]byte
		var walls [2]time.Duration
		for arm, preset := range []string{"", "eight-bit-attention"} {
			request.Preset = preset
			for round := range 2 {
				if arm+round > 0 {
					testcheck.NoError(t, generator.Reset(ctx, request))
				}
				drawn, took := runResidentGeneration(t, ctx, generator, request)
				pictures[arm], walls[arm] = drawn.Data, took
			}
			if generator.request.EightBitAttention != (preset != "") {
				t.Errorf("%s's session given preset %q computes its attention in 8 bits: %t", model.directory, preset, generator.request.EightBitAttention)
			}
		}
		closeReporting(t, ctx, generator)
		t.Logf("%s at %d x %d: %.1f s at full precision, %.1f s with 8-bit attention, %.2fx",
			model.directory, model.side, model.side, walls[0].Seconds(), walls[1].Seconds(), walls[0].Seconds()/walls[1].Seconds())
		if bytes.Equal(pictures[0], pictures[1]) {
			t.Errorf("%s drew the full-precision picture under the preset, so its attention was not the 8-bit kernel's", model.directory)
		}
		if walls[1] >= walls[0] {
			t.Errorf("%s drew in %s with 8-bit attention, no faster than %s at full precision", model.directory, walls[1], walls[0])
		}
	}
}
