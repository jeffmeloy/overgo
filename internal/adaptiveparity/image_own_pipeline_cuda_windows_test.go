//go:build windows

package adaptiveparity

import (
	"bytes"
	"encoding/json"
	"image/png"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// ownPictureScript runs the image model's own pipeline and writes the fixture.
const ownPictureScript = "scripts/python_golden_references/krea_own_picture.py"

// ownPictures is what the image model's own pipeline drew: the pipeline its
// card names, loaded from the model's directory and called as the card calls
// it, for each prompt and seed at the size and steps given. TextEncoder says
// the arithmetic its text encoder ran in. Means holds each picture as the mean
// of every colour over squares of Square pixels, a byte each, row by row.
type ownPictures struct {
	Original    string `json:"original"`
	TextEncoder string `json:"text_encoder"`
	Torch       string `json:"torch"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	Steps       int    `json:"steps"`
	Square      int    `json:"square"`
	Pictures    []struct {
		Name   string `json:"name"`
		Prompt string `json:"prompt"`
		Seed   int64  `json:"seed"`
		Means  []byte `json:"means"`
	} `json:"pictures"`
}

// ownPictureFloor is the agreement, in dB over the fixture's squares, a served
// picture is held to against the model's own. The served picture agrees with
// it at 34.7 dB. A picture whose closing template tokens are numbered past the
// padding stands at 18 dB, one drawn besides at the timestep shift its size
// alone would give at 16.5, and the model's own picture from another seed at
// 15.
const ownPictureFloor = 31

// pictureService is the way a request for a picture is served from a model's
// directory: the image capability, a candidate execution of the recipe the
// directory resolves to, and a store that holds what that recipe binds.
type pictureService struct {
	store      *overgodb.Store
	capability mediacapability.Capability
	execution  modelrecipe.CapabilityEvidenceSelection
	directory  string
}

// pictureServing serves pictures from an image model's directory on an empty
// store, which is given the model's inventory, its components' manifests and
// its profile.
func pictureServing(t *testing.T, directory, name string) pictureService {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	capability := mediacapability.Catalog[recipe.TaskImageGen]
	source, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	definition, facts, err := source.Define(source.Inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := source.Inventory.Batch("test/" + name + "/inventory")
	if err != nil {
		t.Fatal(err)
	}
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	batch.Contents = append(batch.Contents, facts...)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	return pictureService{store: store, capability: capability, execution: modelrecipetest.CandidateExecution(t, store, program), directory: directory}
}

// agreement serves one request for a picture and returns how nearly the
// picture agrees with a reference's picture of the same request, in dB over
// the means of every colour in squares of side pixels, which the reference
// gives a byte each, row by row.
func (service pictureService) agreement(t *testing.T, request any, width, height, side int, reference []byte) float64 {
	t.Helper()
	raw, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	output, err := service.capability.Execute(t.Context(), service.store, service.directory, service.execution, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	served, ok := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
	if !ok {
		t.Fatalf("a served picture is %T", capabilityruntime.Unwrap(output))
	}
	decoded, err := png.Decode(bytes.NewReader(served.Data))
	if err != nil {
		t.Fatal(err)
	}
	means := squareMeans(decoded, side)
	if bounds := decoded.Bounds(); bounds.Dx() != width || bounds.Dy() != height || len(means) != len(reference) {
		t.Fatalf("served %v, %d square means; the reference is %dx%d, %d", bounds, len(means), width, height, len(reference))
	}
	var squared float64
	for index, mean := range means {
		apart := mean - float64(reference[index])
		squared += apart * apart
	}
	return 20 * math.Log10(math.MaxUint8/math.Sqrt(squared/float64(len(means))))
}

// TestServedPictureMatchesTheModelsOwnPipeline holds the picture a request is
// served to the image model's own pipeline: the same prompt, seed, size and
// steps, and nothing else named, as a page sends them. The runtime under the
// request is held stage by stage to the earlier implementation; this is the
// path around it, what a request that names no schedule resolves to and what
// the text encoder is given, and the reference is the model's own code and
// not an earlier port of it.
//
// The reference's text encoder is run in float32. In bfloat16 the pipeline's
// own text encoder stands 1.5 parts in 100 from its float32 states, and that
// alone moves its picture to 21 dB from the float32 one: the served text
// encoder, bfloat16 weights under float32 activations, stands a third of a
// part in 100 from them, so the float32 picture is the one a faithful port is
// near.
func TestServedPictureMatchesTheModelsOwnPipeline(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Krea-2-Turbo")
	var own ownPictures
	if err := jsonfile.DecodeStrict(filepath.Join(testutil.RepoRoot(t), "fixtures", "krea", "own_picture.json"), &own); err != nil {
		t.Fatalf("the model's own picture is absent (%v); write it with %s", err, ownPictureScript)
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	service := pictureServing(t, directory, "own-picture")
	for _, picture := range own.Pictures {
		agreement := service.agreement(t, latentimage.Request{Prompt: picture.Prompt, Width: own.Width, Height: own.Height, Steps: own.Steps, Seed: picture.Seed},
			own.Width, own.Height, own.Square, picture.Means)
		t.Logf("%s, seed %d: the served picture against the model's own pipeline's, its text encoder in %s, %.2f dB over %d-pixel squares",
			picture.Name, picture.Seed, own.TextEncoder, agreement, own.Square)
		if agreement < ownPictureFloor {
			t.Errorf("%s: the served picture agrees with the model's own pipeline's at %.2f dB over %d-pixel squares, below %d", picture.Name, agreement, own.Square, ownPictureFloor)
		}
	}
}
