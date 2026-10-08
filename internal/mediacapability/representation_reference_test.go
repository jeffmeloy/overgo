//go:build modeltest

package mediacapability

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/fixtureasset"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/rae"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// roundTripScript writes the fixture this test reads, from the autoencoder's
// own code. It is kept where Python is kept, on the machine that ran it and
// not in git: the fixture is the data.
const roundTripScript = "scripts/python_golden_references/rae_round_trip.py"

const (
	// roundTripTolerance is how far a picture that came back may stand from
	// the reference's, in the signed-unit range a clip is held in: what the
	// decoder alone is held to in [0, 1], which is twice as wide a range in
	// these units. The encoder before it adds no more; measured, the two
	// joined are within a tenth of it.
	roundTripTolerance = 2e-3
	// oneLevel is one of a pixel's 255 steps in the signed-unit range: what a
	// frame cut by another build of ffmpeg may differ by.
	oneLevel = 2.0 / 255
)

// roundTripTensor names one tensor of the fixture: its raw file, its shape
// and the digest of its bytes.
type roundTripTensor struct {
	File   string `json:"file"`
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
}

// roundTripReference is fixtures/rae/round_trip.json: one frame of the
// reconstruction evaluation as the evaluation hands it to an autoencoder, and
// the picture the reference autoencoder makes of it.
type roundTripReference struct {
	Source     string `json:"source"`
	Commit     string `json:"reference_commit"`
	Encoder    string `json:"encoder"`
	Decoder    string `json:"decoder"`
	Statistics string `json:"statistics"`
	Case       string `json:"case"`
	Clip       string `json:"clip"`
	ClipSHA256 string `json:"clip_sha256"`
	Cut        struct {
		Frames int  `json:"frames"`
		FPS    int  `json:"fps"`
		Width  int  `json:"width"`
		Height int  `json:"height"`
		Crop   bool `json:"crop"`
	} `json:"cut"`
	LatentShape      []int           `json:"latent_shape"`
	MeanSquaredError float64         `json:"mean_squared_error"`
	Picture          roundTripTensor `json:"picture"`
	Image            roundTripTensor `json:"image"`
}

// TestRepresentationRoundTripMatchesReference holds the round trip a
// reconstruction evaluation scores to the autoencoder's own code. The two
// halves are each held to a reference on their own, the encoder on a
// synthetic picture and the decoder on a seeded latent; nothing held them
// joined, on a picture an evaluation asks about, through the autoencoder an
// evaluation opens. Here one frame of the evaluation's first clip, cut as the
// evaluation cuts it, goes through that autoencoder in the signed-unit range
// a clip is held in, and every pixel that comes back is held to the picture
// the reference's forward pass makes of the same frame. The error the
// evaluation would score is then the reference's own.
//
// Where the clip and ffmpeg are on the machine the frame is cut again and
// held to the fixture's, so the picture the reference was given is the one
// the evaluation scores.
func TestRepresentationRoundTripMatchesReference(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	fixtures := filepath.Join(root, "fixtures", "rae")
	var reference roundTripReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "round_trip.json"), &reference); err != nil {
		t.Fatalf("the reference round trip is absent (%v); write it with %s", err, roundTripScript)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	directory := roots.ResolveModelPath("RAE-collections")
	collection, held, err := rae.ResolveCollection(directory)
	if err != nil || !held {
		testskip.NotApplicable(t, "no RAE collection with its encoder is at "+directory)
	}
	if _, err := os.Stat(filepath.Join(collection.EncoderDirectory(), "model.safetensors")); err != nil {
		testskip.NotApplicable(t, "the collection's encoder weights are absent: "+collection.EncoderDirectory())
	}
	testskip.Short(t, "loads the collection's 1.7 GB decoder")
	cut := reference.Cut
	elements := media.RGBChannels * cut.Frames * cut.Height * cut.Width
	picture, err := fixtureasset.LoadF32(fixtures, reference.Picture.File, reference.Picture.SHA256, elements)
	if err != nil {
		t.Fatal(err)
	}
	want, err := fixtureasset.LoadF32(fixtures, reference.Image.File, reference.Image.SHA256, elements)
	if err != nil {
		t.Fatal(err)
	}
	autoencoder, err := OpenAutoencoder(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer autoencoder.Close(t.Context())
	source := latentvideo.SourceVideo{Pixels: picture, Channels: media.RGBChannels, Frames: cut.Frames, Height: cut.Height, Width: cut.Width}
	output, latent, err := autoencoder.RoundTrip(t.Context(), source)
	if err != nil {
		t.Fatal(err)
	}
	if shape := reference.LatentShape; len(shape) != 3 || latent.Channels != shape[0] || latent.Height != shape[1] || latent.Width != shape[2] {
		t.Fatalf("the latent is %d channels over %d by %d, and the reference's is %v", latent.Channels, latent.Height, latent.Width, reference.LatentShape)
	}
	if len(output.Pixels) != len(want) {
		t.Fatalf("%d values came back, and the reference's picture has %d", len(output.Pixels), len(want))
	}
	worst, squared := 0.0, 0.0
	for index, value := range output.Pixels {
		// The reference's picture is in [0, 1]; a clip is held in [-1, 1].
		worst = max(worst, math.Abs(float64(value)-(2*float64(want[index])-1)))
		// The evaluation's error, put in [0, 1] as the reference states its own.
		difference := (float64(value) - float64(picture[index])) / 2
		squared += difference * difference
	}
	scored := squared / float64(len(want))
	t.Logf("%s: the largest difference of a pixel from the reference's is %.3g; the round trip's mean squared error is %.6f and the reference's %.6f",
		reference.Case, worst, scored, reference.MeanSquaredError)
	if worst > roundTripTolerance {
		t.Errorf("a pixel that came back is %.3g from the reference's, beyond %.3g", worst, roundTripTolerance)
	}
	if math.Abs(scored-reference.MeanSquaredError) > roundTripTolerance*reference.MeanSquaredError {
		t.Errorf("the round trip's error is %.6f and the reference's %.6f", scored, reference.MeanSquaredError)
	}

	clip := filepath.Join(roots.Datasets, "VidGen", reference.Clip)
	ffmpeg, err := media.ResolveFFmpeg(os.Getenv("OVERGO_FFMPEG"))
	if _, missing := os.Stat(clip); missing != nil || err != nil {
		t.Logf("the clip or ffmpeg is not on this machine (%v, %v): the fixture's frame is not cut again", missing, err)
		return
	}
	decode := media.DecodeVideoFileAt
	if cut.Crop {
		decode = media.DecodeVideoFileCovering
	}
	frames, err := decode(t.Context(), clip, ffmpeg, float64(cut.FPS), cut.Frames, cut.Width, cut.Height)
	if err != nil {
		t.Fatal(err)
	}
	again, err := latentvideo.SourceVideoFromFrames(frames)
	if err != nil || len(again.Pixels) != len(picture) {
		t.Fatalf("the clip cut again holds %d values (%v), and the fixture's frame %d", len(again.Pixels), err, len(picture))
	}
	apart := 0.0
	for index, value := range again.Pixels {
		apart = max(apart, math.Abs(float64(value)-float64(picture[index])))
	}
	if apart > oneLevel {
		t.Errorf("the frame the evaluation cuts is %.3g from the frame the reference was given, more than one level of a pixel", apart)
	}
	t.Logf("the frame cut again from %s is within %.3g of the fixture's", reference.Clip, apart)
}
