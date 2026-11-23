//go:build windows

package latentimage

import (
	"bytes"
	"math"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestAFlowPictureIsSharpenedAsMOSSsIs holds sharpen-any-model/do: a flow profile's request (Krea's) asking a
// sharpening is no rotary-only ask, and a flow picture's decoded pixels, in [-1, 1], are finished as MOSS's are: the
// sharpened picture is their unit form sharpened, as ComfyUI's ImageSharpen reads it, and the picture drawn beside it,
// like the picture the request without sharpening gives, is written as a flow profile writes it (encodePNG). Request
// rotaryOnly had counted a sharpening among what only a rotary profile runs, so a Krea request asking one was refused,
// though the finish reads the pixels alone.
func TestAFlowPictureIsSharpenedAsMOSSsIs(t *testing.T) {
	t.Parallel()
	sharpening := media.Sharpening{Radius: 1, Sigma: 0.6, Alpha: 0.7}
	if (Request{Prompt: "a red fox", Width: 64, Height: 48, Steps: 4, Sharpen: &sharpening}).rotaryOnly() {
		t.Fatal("a flow request asking a sharpening is refused as rotary-only")
	}
	const height, width = 48, 64
	pixels, unit := make([]float32, height*width*media.RGBChannels), make([]float32, height*width*media.RGBChannels)
	for index := range pixels {
		pixels[index] = float32(0.9 * math.Sin(float64(index)/6.1) * math.Cos(float64(index)/83))
		unit[index] = (pixels[index] + 1) / 2
	}
	finished, drawn, err := finishedPNGs(pixels, height, width, &sharpening, true)
	testcheck.NoError(t, err)
	plain := testcheck.Must(encodePNG(pixels, height, width))(t)
	if !bytes.Equal(drawn.Data, plain.Data) {
		t.Error("the flow picture drawn beside the sharpened one is not the picture its request gives unsharpened")
	}
	sharpened := testcheck.Must(media.Sharpen(unit, height, width, sharpening))(t)
	if want := testcheck.Must(encodeUnitPNG(sharpened, height, width))(t); !bytes.Equal(finished.Data, want.Data) || bytes.Equal(finished.Data, plain.Data) {
		t.Error("the sharpened flow picture is not its unit pixels sharpened")
	}
	unsharpened, none, err := finishedPNGs(pixels, height, width, nil, true)
	testcheck.NoError(t, err)
	if !bytes.Equal(unsharpened.Data, plain.Data) || none.Data != nil {
		t.Error("a flow picture finished with no sharpening is not its picture alone")
	}
}
