//go:build windows

package adaptiveparity

import (
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// ownGenerationScript runs the video model's own generation and writes the fixture.
const ownGenerationScript = "scripts/python_golden_references/wan_own_generation.py"

// ownGeneration is what the video model's own code drew: its repository's
// generate() for each clip's prompt, seed, frames and guidance scale, at the
// size, steps and shift given. TextEncoder and Model say the arithmetic each
// was held in. Means holds each frame as the mean of every colour over
// squares of Square pixels, a byte each, frame by frame, row by row.
type ownGeneration struct {
	Original    string  `json:"original"`
	Attention   string  `json:"attention"`
	Torch       string  `json:"torch"`
	Solver      string  `json:"solver"`
	TextEncoder string  `json:"text_encoder"`
	Model       string  `json:"model"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	Steps       int     `json:"steps"`
	Shift       float64 `json:"shift"`
	Square      int     `json:"square"`
	Clips       []struct {
		Name       string  `json:"name"`
		Prompt     string  `json:"prompt"`
		Seed       int64   `json:"seed"`
		Frames     int     `json:"frames"`
		GuideScale float64 `json:"guide_scale"`
		Means      []byte  `json:"means"`
	} `json:"clips"`
}

// ownGenerationFloor is the agreement, in dB over the fixture's squares, a
// served clip is held to against the model's own. The served clip agrees with
// it at 45 dB. The model's own clip at a shift of 3 stands at 29 dB from its
// own at 5, with a blank negative prompt at 24, at a guidance scale of 4 for
// 5 at 21, and from another seed at 12.
const ownGenerationFloor = 38

// TestServedGenerationMatchesTheModelsOwnClip holds the clip a text-to-video
// request is served to the video model's own code: the same prompt, seed,
// frames, size, steps, shift and guidance, and no negative prompt named, so
// that each takes the model's own. The runtime under the request is held to
// the earlier implementation; this is the path a request takes around it, the
// text read into its two contexts, the noise drawn for the seed, the sampler
// and its schedule, the guidance, the decoding and the clip written.
//
// The reference is run in float32. The repository holds its text encoder and
// its video model in bfloat16, and its own clip in float32 stands 34 dB from
// its own in bfloat16: the served clip, bfloat16 weights under float32
// activations, stands 45 dB from the float32 one and 33 from the bfloat16.
//
// The clip is drawn where the model's own clip is steady, nine frames with
// guidance. It is not steady everywhere. At the settings the model's held-out
// reading is drawn at, seventeen frames with a guidance scale of 5, its own
// two precisions stand 16 to 21 dB apart and the served clip 16 to 23 dB from
// either, where a clip from another seed stands at 10: there a comparison of
// squares tells the served clip from another clip, and not from the model's
// own rounding. Seventeen frames without guidance are steady again, and the
// served clip stands 46 dB from the model's float32 one.
func TestServedGenerationMatchesTheModelsOwnClip(t *testing.T) {
	cudatest.Require(t)
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	var own ownGeneration
	if err := jsonfile.DecodeStrict(filepath.Join(testutil.RepoRoot(t), "fixtures", "wan", "own_generation.json"), &own); err != nil {
		t.Fatalf("the model's own clip is absent (%v); write it with %s", err, ownGenerationScript)
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "wan-own-generation")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profile.ID)
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
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/own-generation", Artifacts: []artifact.Descriptor{{ID: modelID, Size: directoryExtent(t, wan)}}, Contents: []artifact.Content{profileContent},
	}); err != nil {
		t.Fatal(err)
	}
	execution := modelrecipetest.CandidateExecution(t, store, program)
	for _, clip := range own.Clips {
		raw, err := json.Marshal(latentvideo.WanRequest{
			Prompt: clip.Prompt, Seed: uint64(clip.Seed), Frames: clip.Frames, Width: own.Width, Height: own.Height,
			Steps: own.Steps, Shift: own.Shift, GuideScale: clip.GuideScale,
		})
		if err != nil {
			t.Fatal(err)
		}
		output, err := mediacapability.Catalog[recipe.TaskVideoGen].Execute(t.Context(), store, wan, execution, string(raw))
		if err != nil {
			t.Fatal(err)
		}
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("a served clip is %T", capabilityruntime.Unwrap(output))
		}
		served := servedSquareMeans(t, video, own.Square)
		if len(served) != clip.Frames || len(clip.Means) != clip.Frames*len(served[0]) {
			t.Fatalf("%s: the served clip is %d frames of %d square means, the model's own %d bytes for %d frames", clip.Name, len(served), len(served[0]), len(clip.Means), clip.Frames)
		}
		var squared float64
		for frame, means := range served {
			for index, mean := range means {
				apart := mean - float64(clip.Means[frame*len(means)+index])
				squared += apart * apart
			}
		}
		agreement := 20 * math.Log10(math.MaxUint8/math.Sqrt(squared/float64(len(clip.Means))))
		t.Logf("%s, seed %d, %d frames at a guidance scale of %g: the served clip against the model's own, its text encoder in %s and its model in %s, %.2f dB over %d-pixel squares",
			clip.Name, clip.Seed, clip.Frames, clip.GuideScale, own.TextEncoder, own.Model, agreement, own.Square)
		if agreement < ownGenerationFloor {
			t.Errorf("%s: the served clip agrees with the model's own code at %.2f dB over %d-pixel squares, below %d", clip.Name, agreement, own.Square, ownGenerationFloor)
		}
	}
}
