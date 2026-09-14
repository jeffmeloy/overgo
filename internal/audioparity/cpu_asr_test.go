package audioparity

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestCPUASRReference is the mandatory real-artifact acceptance door. Missing
// store bindings, model, corpus, capture or any numerical boundary fail it.
// Python and GPU execution are not prerequisites: the pinned CPU capture is.
func TestCPUASRReference(t *testing.T) {
	t.Parallel()
	storeRoot := resolveReferenceRoots(t).store
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	encoded, err := os.ReadFile("testdata/granite_speech_5_election.json")
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("testdata/encoder_capture.json")
	if err != nil {
		t.Fatal(err)
	}
	var capture encoderCapture
	if err = strictjson.DecodeBytes(encoded, &capture); err != nil {
		t.Fatal(err)
	}
	if err = capture.validate(election); err != nil {
		t.Fatal(err)
	}
	golden, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-encoder-golden/v1"), capture)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.RequireTypedContent(t.Context(), store, golden.Descriptor.ID); err != nil {
		t.Fatal(err)
	}
	modelRoot, weights := loadASRModel(t, store, election)
	corpusPath, err := artifact.AvailablePath(t.Context(), store, election.Dataset.Artifact, artifact.LocationFile)
	if err != nil {
		t.Fatal(err)
	}
	verifyASRFile(t, corpusPath, election.Dataset.Artifact, election.Dataset.Bytes)
	tensorID, err := artifact.ParseID("file:sha256:" + capture.TensorsSHA256)
	if err != nil {
		t.Fatal(err)
	}
	tensorPath, err := artifact.AvailablePath(t.Context(), store, tensorID, artifact.LocationFile)
	if err != nil {
		t.Fatal(err)
	}
	oracle, err := safetensors.OpenSource(filepath.Dir(tensorPath))
	if err != nil {
		t.Fatal(err)
	}
	defer oracle.Close()
	digests, err := oracle.ShardDigests()
	if err != nil {
		t.Fatal(err)
	}
	if len(digests) != 1 || digests[0].Name != filepath.Base(tensorPath) || hex.EncodeToString(digests[0].Digest[:]) != capture.TensorsSHA256 || len(oracle.Tensors) != capture.TensorCount {
		t.Fatal("oracle tensor identity differs")
	}
	encoded, err = os.ReadFile("recipes/granite5asr_execution.json")
	if err != nil {
		t.Fatal(err)
	}
	var declaration speechrecognition.Declaration
	if err = strictjson.DecodeBytes(encoded, &declaration); err != nil {
		t.Fatal(err)
	}
	// Explicit integration-test host ceiling, not a production default.
	const referenceMemoryBytes = 4 << 30
	encoder, err := speechrecognition.LoadEncoder(t.Context(), weights, declaration, referenceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("../audiodsp/testdata/power_logmel_frontend.json")
	if err != nil {
		t.Fatal(err)
	}
	var frontendConfig audiodsp.FrontendConfig
	if err = json.Unmarshal(encoded, &frontendConfig); err != nil {
		t.Fatal(err)
	}
	frontend, err := audiodsp.NewFrontend(frontendConfig, referenceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	if !recipe.Preprocessor.Deltas || recipe.Preprocessor.DeltaWinLength%2 != 1 {
		t.Fatal("unsupported processor declaration")
	}
	grouping := audiodsp.GroupedFeatureConfig{StackFrames: int(recipe.Preprocessor.StackFactor), DeltaRadius: int(recipe.Preprocessor.DeltaWinLength-1) / 2, FinalFrameSamples: 1}
	var fw audiodsp.Workspace
	var ew speechrecognition.Workspace
	completed, totalFrames, totalTokens := 0, 0, 0
	err = dataset.ReadParquetTextRows(t.Context(), corpusPath, "bytes", len(capture.Cases), func(index uint64, payload string) error {
		if index >= uint64(len(capture.Cases)) {
			return fmt.Errorf("extra corpus row")
		}
		tc := capture.Cases[index]
		t.Run(tc.Fixture, func(t *testing.T) {
			id, err := artifact.IdentifyBytes(artifact.KindFile, []byte(payload))
			if err != nil || id != tc.Audio {
				t.Fatal("encoded audio identity differs")
			}
			audio, _, err := media.DecodeAudio(t.Context(), []byte(payload), tc.Samples)
			if err != nil {
				t.Fatal(err)
			}
			if uint64(len(audio.Samples)) != tc.Samples || audio.Format.SampleRate != uint64(tc.SampleRate) || audio.Format.Channels != 1 {
				t.Fatal("decoded audio geometry differs")
			}
			wave := captureF32(t, oracle, tc.Fixture+"/waveform", int(tc.Samples))
			if !slices.Equal(audio.Samples, wave) {
				t.Fatal("Go decoded PCM differs from reference")
			}
			features, frames, width, err := frontend.ProcessGrouped(t.Context(), audio.Samples, int(tc.SampleRate), &fw, grouping)
			if err != nil {
				t.Fatal(err)
			}
			if frames != tc.Frames[1] || width != tc.Frames[2] {
				t.Fatal("processor shape differs")
			}
			assertASRBoundary(t, "processor", features, captureF32(t, oracle, tc.Fixture+"/features", frames, width), frontendConfig.FFTLength)
			mask := captureIDs(t, oracle, tc.Fixture+"/mask", frames)
			for _, valid := range mask {
				if valid != 1 {
					t.Fatal("single-clip encoder cannot claim padded-batch parity")
				}
			}
			seen := 0
			start := time.Now()
			hidden, outFrames, err := encoder.Encode(t.Context(), features, frames, &ew, func(trace speechrecognition.Trace) error {
				if trace.Block != seen-1 {
					return fmt.Errorf("missing or reordered encoder boundary")
				}
				name := fmt.Sprintf("layer_%d", trace.Block)
				if trace.Block < 0 {
					name = "input_projection"
				}
				assertASRBoundary(t, name, trace.Values, captureF32(t, oracle, tc.Fixture+"/"+name, trace.Frames, trace.Width), trace.Width)
				seen++
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			logits, err := encoder.Project(t.Context(), hidden, outFrames, &ew)
			if err != nil {
				t.Fatal(err)
			}
			if seen != len(declaration.Blocks)+1 || outFrames <= 0 || len(hidden) == 0 {
				t.Fatal("incomplete encoder boundaries")
			}
			vocab := int(recipe.Config.VocabSize)
			assertASRBoundary(t, "logits", logits, captureF32(t, oracle, tc.Fixture+"/logits", outFrames, vocab), len(hidden)/outFrames)
			frameIDs := make([]int, outFrames)
			ids, err := speechrecognition.GreedyCTC(t.Context(), frameIDs, make([]int, outFrames), logits, outFrames, vocab, int(recipe.Config.PadTokenID))
			if err != nil {
				t.Fatal(err)
			}
			if len(ids) == 0 || !slices.Equal(frameIDs, captureIDs(t, oracle, tc.Fixture+"/tokens", outFrames)) {
				t.Fatal("CTC frame IDs differ or transcript is empty")
			}
			text := tokenizer.Decode(ids)
			if text != tc.Text {
				t.Fatalf("transcript %q, want %q", text, tc.Text)
			}
			retainedHidden, retainedLogits := slices.Clone(hidden), slices.Clone(logits)
			againHidden, againFrames, err := encoder.Encode(t.Context(), features, frames, &ew, nil)
			if err != nil {
				t.Fatal(err)
			}
			againLogits, err := encoder.Project(t.Context(), againHidden, againFrames, &ew)
			if err != nil || againFrames != outFrames || !slices.Equal(retainedHidden, againHidden) || !slices.Equal(retainedLogits, againLogits) {
				t.Fatalf("unobserved replay is not bit-identical: %v", err)
			}
			t.Logf("exact PCM, %d frame IDs, %d collapsed IDs and transcript; %d approximate encoder boundaries; observed forward+comparison wall=%s (not a benchmark)", outFrames, len(ids), seen+1, time.Since(start))
			completed++
			totalFrames += outFrames
			totalTokens += len(ids)
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if completed != len(capture.Cases) || totalFrames == 0 || totalTokens == 0 {
		t.Fatal("incomplete real-artifact acceptance")
	}
	t.Logf("CPU ASR: %d/%d recordings, %d CTC frames, %d transcript tokens; full processor, every layer, logits, IDs and text checked; no training, serving, streaming, WER suite or GPU execution", completed, len(capture.Cases), totalFrames, totalTokens)
}

func verifyASRFile(t *testing.T, path string, want artifact.ID, size uint64) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	id, n, err := artifact.Identify(want.Kind(), file)
	if err != nil || id != want || n != size {
		t.Fatalf("artifact differs: %s (%v)", path, err)
	}
}

func captureF32(t *testing.T, source *safetensors.Source, name string, shape ...int) []float32 {
	t.Helper()
	tensor, ok := source.Tensors[name]
	if !ok || tensor.DType != "F32" || len(shape) != len(tensor.Shape) {
		t.Fatalf("missing or non-F32 tensor %q", name)
	}
	for i, n := range shape {
		if uint64(n) != tensor.Shape[i] {
			t.Fatalf("shape differs for %q", name)
		}
	}
	values, err := safetensors.ReadF32(tensor)
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func captureIDs(t *testing.T, source *safetensors.Source, name string, shape ...int) []int {
	t.Helper()
	tensor, ok := source.Tensors[name]
	wantShape := make([]uint64, len(shape))
	for i, n := range shape {
		if n <= 0 {
			t.Fatal("invalid expected ID shape")
		}
		wantShape[i] = uint64(n)
	}
	if !ok || tensor.DType != "I64" || !slices.Equal(tensor.Shape, wantShape) {
		t.Fatalf("missing or wrong ID tensor %q", name)
	}
	count, valid := checked.Int(tensor.Elements())
	if !valid {
		t.Fatal("captured ID extent overflows")
	}
	data, err := io.ReadAll(tensor.Reader())
	if err != nil {
		t.Fatal(err)
	}
	values := make([]int, count)
	for i := range values {
		value := binary.LittleEndian.Uint64(data[i*8:])
		values[i] = int(value)
		if values[i] < 0 || uint64(values[i]) != value {
			t.Fatal("invalid captured ID")
		}
	}
	return values
}

func assertASRBoundary(t *testing.T, name string, got, want []float32, reduction int) {
	t.Helper()
	maximum := 1.0
	for _, v := range want {
		maximum = max(maximum, math.Abs(float64(v)))
	}
	// One float32 unit roundoff per declared channel/FFT reduction, scaled to
	// the boundary magnitude. This is a conservative comparison protocol,
	// not a proof of accumulated nonlinear-network conditioning.
	tolerance := float64(math.Nextafter32(1, 2)-1) * float64(reduction) * maximum
	testutil.RequireWithin(t, name, got, want, tolerance)
}
