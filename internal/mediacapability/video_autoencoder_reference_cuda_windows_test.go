//go:build windows && modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// videoAutoencoderScript runs the video autoencoder's own code and writes the fixture.
const videoAutoencoderScript = "scripts/python_golden_references/wan_vae_round_trip.py"

const (
	// A served frame's error against its source may stand this far, as a share,
	// from the reference's. The reference's is taken in float; a served frame is
	// bytes, and rounding to a byte adds a twelfth of a level squared, which is
	// counted in before the two are compared. The served frame's error stands
	// 1.6 parts in 100 below it.
	videoAutoencoderErrorShare = 0.04
	// A served value may stand this many of its 255 levels from the reference's
	// rounded one. Half the frame's values stand one level off and none stands
	// further.
	videoAutoencoderLevels = 2
)

// videoAutoencoderReference is fixtures/wan/vae_round_trip.json: one frame of
// the reconstruction evaluation as the evaluation cuts it, and the frame the
// autoencoder's own code makes of it.
type videoAutoencoderReference struct {
	Source     string `json:"source"`
	Weights    string `json:"weights"`
	Case       string `json:"case"`
	Clip       string `json:"clip"`
	ClipSHA256 string `json:"clip_sha256"`
	Cut        struct {
		Frames int  `json:"frames"`
		FPS    int  `json:"fps"`
		Width  int  `json:"width"`
		Height int  `json:"height"`
		Crop   bool `json:"crop,omitempty"`
	} `json:"cut"`
	LatentShape      []int           `json:"latent_shape"`
	MeanSquaredError float64         `json:"mean_squared_error"`
	PSNR             float64         `json:"psnr_db"`
	Picture          roundTripTensor `json:"picture"`
	Image            roundTripTensor `json:"image"`
}

// TestVideoAutoencoderRoundTripMatchesItsOwnCode holds the reconstruction a
// request is served, through the catalog's capability on the installed Wan
// autoencoder, to the autoencoder's own code: one frame of the reconstruction
// evaluation's first clip, cut as the evaluation cuts it, sent as the PNG a
// request carries, and the frame that comes back held pixel for pixel to the
// frame the model's own decode(encode(x)) makes of it, and its error against
// the source to the reference's own. A reconstruction measure compares an
// output with its input, so a path that lost detail would read as a model
// that loses detail; here the error an evaluation scores is the model's own.
func TestVideoAutoencoderRoundTripMatchesItsOwnCode(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	fixtures := filepath.Join(testutil.RepoRoot(t), "fixtures", "wan")
	var reference videoAutoencoderReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "vae_round_trip.json"), &reference); err != nil {
		t.Fatalf("the reference round trip is absent (%v); write it with %s", err, videoAutoencoderScript)
	}
	read := func(tensor roundTripTensor) (image.Image, []byte) {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(fixtures, filepath.FromSlash(tensor.File)))
		if err != nil {
			t.Fatal(err)
		}
		picture, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		return picture, data
	}
	source, sourceBytes := read(reference.Picture)
	want, _ := read(reference.Image)
	capability := Catalog[recipe.TaskReconstruction]
	resolved, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	modelID := resolved.Inventory.Manifest.ID
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskReconstruction, modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishArtifactExtent(t, store, modelID, latentvideo.AutoencoderWeights(directory))
	request, err := json.Marshal(latentvideo.ReconstructionRequest{Image: sourceBytes})
	if err != nil {
		t.Fatal(err)
	}
	output, err := capability.Execute(t.Context(), store, directory, modelrecipetest.CandidateExecution(t, store, program), string(request))
	if err != nil {
		t.Fatal(err)
	}
	reconstruction, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
	if !ok {
		t.Fatalf("reconstruction output type=%T", output)
	}
	served, err := png.Decode(bytes.NewReader(reconstruction.Data))
	if err != nil {
		t.Fatal(err)
	}
	if served.Bounds() != source.Bounds() || want.Bounds() != source.Bounds() {
		t.Fatalf("the frame is %v, the served one %v and the reference's %v", source.Bounds(), served.Bounds(), want.Bounds())
	}
	bounds := source.Bounds()
	var squared float64
	var worst, apart int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			sourceRed, sourceGreen, sourceBlue, _ := source.At(x, y).RGBA()
			servedRed, servedGreen, servedBlue, _ := served.At(x, y).RGBA()
			wantRed, wantGreen, wantBlue, _ := want.At(x, y).RGBA()
			for _, colour := range [][3]uint32{{sourceRed, servedRed, wantRed}, {sourceGreen, servedGreen, wantGreen}, {sourceBlue, servedBlue, wantBlue}} {
				from, got, held := int(colour[0]>>8), int(colour[1]>>8), int(colour[2]>>8)
				squared += float64((got - from) * (got - from))
				if off := max(got-held, held-got); off > 0 {
					apart++
					worst = max(worst, off)
				}
			}
		}
	}
	values := float64(3 * bounds.Dx() * bounds.Dy())
	// In the signed-unit range a frame is held in, where a level is 2 of 255.
	const level = 2.0 / math.MaxUint8
	scored := squared / values * level * level
	rounded := reference.MeanSquaredError + level*level/12
	t.Logf("%s: the served frame's error is %.6g (%.3f dB), the reference's %.6g in float and %.6g once rounded to bytes (%.3f dB); %d of %.0f values differ from the reference's rounded frame, the furthest by %d levels",
		reference.Case, scored, 10*math.Log10(4/scored), reference.MeanSquaredError, rounded, 10*math.Log10(4/rounded), apart, values, worst)
	if math.Abs(scored-rounded) > videoAutoencoderErrorShare*rounded {
		t.Errorf("the served frame's error is %.6g and the reference's, rounded to bytes, %.6g", scored, rounded)
	}
	if worst > videoAutoencoderLevels {
		t.Errorf("a served value is %d levels from the reference's, beyond %d", worst, videoAutoencoderLevels)
	}
}
