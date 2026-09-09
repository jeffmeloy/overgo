package audioparity

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestSpeakerDiarizationAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": pinned real speaker model, independent traces and continuous annotated speech")
	}
	root := cmp.Or(os.Getenv("OVERGO_AUDIO_SPEAKER_REFERENCE"), filepath.Join(testutil.RepoRoot(t), "tmp", "speaker-reference"))
	t.Run("numerical", func(t *testing.T) { verifySpeakerNumerical(t, root) })
	t.Run("stored_and_command", func(t *testing.T) { verifySpeakerStoredSession(t, root) })
}

type speakerCaptureFile struct {
	Path, SHA256 string
	Bytes        uint64
}

func verifySpeakerNumerical(t *testing.T, root string) {
	t.Helper()
	var reference struct {
		Revision            string
		CaptureDriverSHA256 string `json:"capture_driver_sha256"`
		Tolerance           struct {
			Maximum float64 `json:"maximum_absolute_boundary_difference"`
		}
		Files []speakerCaptureFile
	}
	readAudioFixtureJSON(t, "testdata/speaker_reference.json", &reference)
	if reference.Revision != "3497b7cc44753e2c141d8fe60ac42cec433e3281" || len(reference.Files) != 156 || reference.Tolerance.Maximum != 1e-3 {
		t.Fatal("reference authority or predeclared tolerance differs")
	}
	driver, err := os.ReadFile("testdata/speaker_reference_capture.cpp")
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, driver)
	if err != nil || id.DigestHex() != reference.CaptureDriverSHA256 {
		t.Fatal("reference driver identity differs")
	}
	files := make(map[string]speakerCaptureFile, len(reference.Files))
	for _, file := range reference.Files {
		if !filepath.IsLocal(file.Path) || files[file.Path].Path != "" || file.Bytes == 0 || file.Bytes%4 != 0 {
			t.Fatal("invalid capture inventory")
		}
		files[file.Path] = file
	}
	read := func(path string) []float32 {
		file, ok := files[path]
		if !ok {
			t.Fatalf("capture absent: %s", path)
		}
		physical := filepath.Join(root, filepath.FromSlash(path))
		verifyASRFile(t, physical, alignmentFileID(t, file.SHA256), file.Bytes)
		data, err := os.ReadFile(physical)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(data)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
			if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
				t.Fatal("nonfinite oracle")
			}
		}
		return out
	}
	modelRoot := filepath.Join(root, "model")
	verifySpeakerModel(t, modelRoot)
	weights, err := safetensors.OpenSource(modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	var declaration struct {
		Encoder  speechrecognition.Declaration
		Activity speechrecognition.ActivityBinding
	}
	readAudioFixtureJSON(t, "testdata/speaker_activity_declaration.json", &declaration)
	model, err := speechrecognition.LoadSpeakerActivity(t.Context(), weights, declaration.Encoder, declaration.Activity, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := speechrecognition.LoadSpeakerActivity(t.Context(), weights, declaration.Encoder, declaration.Activity, 1); err == nil {
		t.Fatal("weight budget refusal missing")
	}
	var config audiodsp.FrontendConfig
	readAudioFixtureJSON(t, "testdata/speaker_frontend.json", &config)
	frontend, err := audiodsp.NewFrontend(config, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	var workspace speechrecognition.ActivityWorkspace
	var frontendWorkspace audiodsp.Workspace
	var lastFeatures []float32
	lastFrames := 0
	consumed := map[string]bool{}
	for _, name := range []string{"0", "20", "40", "partial"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(root, "clip-"+name+".wav"))
			if err != nil {
				t.Fatal(err)
			}
			if name == "partial" {
				id, err := artifact.IdentifyBytes(artifact.KindFile, data)
				if err != nil || id.DigestHex() != "ff407d029c95a37cd986d6f496c4780c58efa8afe4760ab84150942c25eba8ce" {
					t.Fatal("partial source differs")
				}
			}
			audio, _, err := media.DecodeAudio(t.Context(), data, uint64(len(data)))
			if err != nil {
				t.Fatal(err)
			}
			peak := slices.Max(audio.Samples)
			gain := float32(1) / (peak + float32(1e-3))
			for i := range audio.Samples {
				audio.Samples[i] *= gain
			}
			features, frames, err := frontend.Process(t.Context(), [][]float32{audio.Samples}, config.SampleRate, &frontendWorkspace, audiodsp.ProcessOptions{FrameLimit: len(audio.Samples) / int(config.Geometry.HopSamples)})
			if err != nil {
				t.Fatal(err)
			}
			var maximum float64
			values := 0
			compare := func(path string, got []float32) {
				want := read(path)
				consumed[path] = true
				if path == "capture-partial/features.f32" {
					if len(want) != 2000*declaration.Activity.Bands || len(got) != 1985*declaration.Activity.Bands {
						t.Fatal("partial feature mask geometry differs")
					}
					for _, v := range want[len(got):] {
						if v != 0 {
							t.Fatal("reference padded feature is not masked")
						}
					}
					want = want[:len(got)]
				}
				if len(got) != len(want) {
					t.Fatalf("capture shape %s: %d != %d", path, len(got), len(want))
				}
				var delta float64
				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite prediction")
					}
					delta = max(delta, math.Abs(float64(v-want[i])))
				}
				maximum = max(maximum, delta)
				values += len(got)
				if delta > reference.Tolerance.Maximum {
					t.Fatalf("%s max_abs=%g exceeds %g", path, delta, reference.Tolerance.Maximum)
				}
			}
			prefix := "capture-" + name + "/"
			compare(prefix+"features.f32", features)
			traces := 0
			probabilities, rows, err := model.Predict(t.Context(), features, frames, &workspace, func(trace speechrecognition.Trace) error {
				if trace.Block < 0 {
					return nil
				}
				label := fmt.Sprintf("encoder-%d", trace.Block)
				if trace.Block == len(declaration.Encoder.Blocks) {
					label = "projection"
				} else if trace.Block > len(declaration.Encoder.Blocks) {
					label = fmt.Sprintf("postnorm-%d", trace.Block-len(declaration.Encoder.Blocks)-1)
				}
				compare(prefix+label+".f32", trace.Values)
				traces++
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			compare(prefix+"probabilities.f32", probabilities)
			if traces != len(declaration.Encoder.Blocks)+len(declaration.Activity.Blocks)+1 || rows != (frames+7)/8 {
				t.Fatal("trace or frame denominator differs")
			}
			lastFeatures, lastFrames = features, frames
			t.Logf("waveform-to-probabilities %s: features=%d output=%d traces=%d values=%d max_abs=%g", name, frames, rows, traces, values, maximum)
		})
	}
	if len(consumed) != len(files) {
		t.Fatalf("capture denominator %d/%d", len(consumed), len(files))
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := model.Predict(ctx, lastFeatures, lastFrames, &workspace, nil); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation not preserved")
	}
	other := *model
	if _, _, err := other.Predict(t.Context(), lastFeatures, lastFrames, &workspace, nil); err == nil {
		t.Fatal("foreign workspace accepted")
	}
	var zero speechrecognition.SpeakerActivity
	if _, _, err := zero.Predict(t.Context(), lastFeatures, lastFrames, &workspace, nil); err == nil {
		t.Fatal("zero model accepted")
	}
	refusal := errors.New("observer refuses")
	if _, _, err := model.Predict(t.Context(), lastFeatures, lastFrames, &workspace, func(speechrecognition.Trace) error { return refusal }); !errors.Is(err, refusal) {
		t.Fatal("observer refusal lost")
	}
	lastFeatures[0] = float32(math.NaN())
	if _, _, err := model.Predict(t.Context(), lastFeatures, lastFrames, &workspace, nil); err == nil {
		t.Fatal("nonfinite features accepted")
	}
}
