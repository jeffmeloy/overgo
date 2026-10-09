//go:build windows

package adaptiveparity

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/draw"
	"image/gif"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// ownEditScript runs the edit model's own inference and writes the fixture.
const ownEditScript = "scripts/python_golden_references/liveedit_own_edit.py"

// ownEdit is what the edit model's own code made of clips: its repository's
// inference, run on each clip's frames as Cut names them, with the instruction
// and seed given. Means holds each edited frame as the mean of every colour
// over squares of Square pixels, a byte each, frame by frame, row by row.
type ownEdit struct {
	Original     string `json:"original"`
	Attention    string `json:"attention"`
	Torch        string `json:"torch"`
	Seed         int64  `json:"seed"`
	LatentFrames int    `json:"latent_frames"`
	Frames       int    `json:"frames"`
	FPS          int    `json:"fps"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	Square       int    `json:"square"`
	Cut          string `json:"cut"`
	Clips        []struct {
		Name        string `json:"name"`
		Instruction string `json:"instruction"`
		Means       []byte `json:"means"`
	} `json:"clips"`
}

// ownEditFloor is the agreement, in dB over the fixture's squares, a served
// edit is held to against the model's own. overgo's edit of the fixture's clip
// agrees with it at 38 dB. The model's own code at another seed agrees with
// itself at 26, so an edit drawn from other noise fails; and the same edit
// from a source carried as a GIF agrees at 26.
const ownEditFloor = 34

// TestServedEditMatchesTheModelsOwnEdit holds the edit a request is served to
// the edit model's own code: its repository's inference, run from the
// checkpoint on the same frames with the same instruction and seed. The
// reference case of TestServedEditMatchesTheReferenceEdit is the earlier
// implementation's clip of fruit in a bowl; this one is the model's own code,
// on a clip of 21 frames with a person in it, which is what the held-out
// measure edits. Agreement here says the served edit is the model's edit, what
// it changes and what it keeps: told to recolour a shirt, the model's own code
// draws the man wearing it as another, younger man, and so does the served
// edit, square for square.
func TestServedEditMatchesTheModelsOwnEdit(t *testing.T) {
	cudatest.Require(t)
	edit := modeltest.Directory(t, "LiveEdit")
	wan, checkpoint := latentvideo.LiveEditArtifacts(edit)
	root := testutil.RepoRoot(t)
	var own ownEdit
	if err := jsonfile.DecodeStrict(filepath.Join(root, "fixtures", "liveedit", "own_edit.json"), &own); err != nil {
		t.Fatalf("the model's own edit is absent (%v); write it with %s", err, ownEditScript)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	required := []string{checkpoint, latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint}
	for _, clip := range own.Clips {
		required = append(required, filepath.Join(roots.Datasets, "VidGen", clip.Name+".mp4"))
	}
	for _, path := range required {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("UNAVAILABLE: %v", err)
		}
	}
	ffmpeg, err := media.ResolveFFmpeg(os.Getenv("OVERGO_FFMPEG"))
	if err != nil {
		testskip.NotApplicable(t, "no FFmpeg to cut the clip with")
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	store, serve := editServing(t, edit, wan, "liveedit-own-edit")
	plane := own.Width * own.Height
	for _, clip := range own.Clips {
		// The clip cut as the fixture's Cut says, which is how the evaluation cuts one.
		frames, err := media.DecodeVideoFileAt(t.Context(), filepath.Join(roots.Datasets, "VidGen", clip.Name+".mp4"), ffmpeg, float64(own.FPS), own.Frames, own.Width, own.Height)
		if err != nil || len(frames) != own.Frames {
			t.Fatalf("%s gave %d of %d frames: %v", clip.Name, len(frames), own.Frames, err)
		}
		source := latentvideo.SourceVideo{
			Pixels: make([]float32, media.RGBChannels*own.Frames*plane), Channels: media.RGBChannels,
			Frames: own.Frames, Height: own.Height, Width: own.Width,
		}
		for index, frame := range frames {
			origin := frame.Bounds().Min
			for at := range plane {
				red, green, blue, _ := frame.At(origin.X+at%own.Width, origin.Y+at/own.Width).RGBA()
				for channel, value := range []uint32{red, green, blue} {
					source.Pixels[(channel*own.Frames+index)*plane+at] = float32(value) / math.MaxUint16
				}
			}
		}
		content, err := latentvideo.SourceFramesContent(source)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/own-edit/" + clip.Name, Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		served := servedSquareMeans(t, serve(latentvideo.ReferenceEditRequest{Prompt: clip.Instruction, Seed: own.Seed, SourceArtifact: content.Descriptor.ID}), own.Square)
		if len(served) != own.Frames || len(clip.Means) != own.Frames*len(served[0]) {
			t.Fatalf("%s: the served edit is %d frames of %d square means, the model's own %d bytes for %d frames", clip.Name, len(served), len(served[0]), len(clip.Means), own.Frames)
		}
		var squared float64
		for frame, means := range served {
			for index, mean := range means {
				apart := mean - float64(clip.Means[frame*len(means)+index])
				squared += apart * apart
			}
		}
		agreement := 20 * math.Log10(math.MaxUint8/math.Sqrt(squared/float64(len(clip.Means))))
		t.Logf("%s, %q: the served edit against the model's own, %.2f dB over %d-pixel squares", clip.Name, clip.Instruction, agreement, own.Square)
		if agreement < ownEditFloor {
			t.Errorf("%s: the served edit agrees with the model's own code at %.2f dB over %d-pixel squares, below %d", clip.Name, agreement, own.Square, ownEditFloor)
		}
	}
}

// editServing opens a store that holds what a served edit by the model at edit
// stands on, the base model's profile and the model's artifact at the size of
// its directory, and gives the call that serves one request against it and
// returns the edited clip.
func editServing(t *testing.T, edit, wan, name string) (*overgodb.Store, func(latentvideo.ReferenceEditRequest) latentvideo.EncodedVideo) {
	t.Helper()
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
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
	t.Cleanup(func() { store.Close() })
	var extent uint64
	if err := filepath.WalkDir(edit, func(_ string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		info, err := entry.Info()
		extent += uint64(info.Size())
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "test/artifact/" + modelID.String(), Artifacts: []artifact.Descriptor{{ID: modelID, Size: extent}}, Contents: []artifact.Content{profileContent},
	}); err != nil {
		t.Fatal(err)
	}
	return store, func(form latentvideo.ReferenceEditRequest) latentvideo.EncodedVideo {
		t.Helper()
		request, err := json.Marshal(form)
		if err != nil {
			t.Fatal(err)
		}
		output, err := mediacapability.Catalog[recipe.TaskVideoGen].Execute(t.Context(), store, edit, modelrecipetest.CandidateExecution(t, store, program), string(request))
		if err != nil {
			t.Fatal(err)
		}
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("a served edit is %T", output)
		}
		return video
	}
}

// servedSquareMeans is a served clip, frame by frame, each as the mean of
// every colour over squares of side pixels (squareMeans). A served clip is a
// GIF, dithered to a palette, so single pixels differ by the dither; the mean
// of a square does not.
func servedSquareMeans(t *testing.T, video latentvideo.EncodedVideo, side int) [][]float64 {
	t.Helper()
	animation, err := gif.DecodeAll(bytes.NewReader(video.Data))
	if err != nil {
		t.Fatal(err)
	}
	canvas := image.NewRGBA(image.Rect(0, 0, video.Width, video.Height))
	means := make([][]float64, len(animation.Image))
	for frame, paletted := range animation.Image {
		draw.Draw(canvas, paletted.Bounds(), paletted, paletted.Bounds().Min, draw.Over)
		means[frame] = squareMeans(canvas, side)
	}
	return means
}
