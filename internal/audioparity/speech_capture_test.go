package audioparity

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// overgoMade is the maker a remade capture names in these tests: overgo at a full commit through the remaking command.
var overgoMade = captureProducer{Commit: strings.Repeat("a", 40), Command: SpeechCaptureCommand}

// tinyFrontend is a frontend small enough for a synthetic clip, giving bins mel bands per frame.
func tinyFrontend(t *testing.T, bins uint32) *audiodsp.Frontend {
	t.Helper()
	return testcheck.Must(audiodsp.NewFrontend(audiodsp.FrontendConfig{
		SampleRate: 16000, Geometry: recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 2, FeatureBins: bins},
		FFTLength: 16, FrameSpan: 4, Padding: "zero", Window: "rectangular",
		Mel: audiodsp.MelConfig{Scale: "htk", MinFrequency: 20, MaxFrequency: 7000},
		Log: audiodsp.LogConfig{Power: true, Base: "natural", GuardMode: "clamp", Guard: 1e-8, Scale: 1},
	}, 1<<24))(t)
}

// TestCaptureTensorsAreWrittenAsTheAcceptancesReadThem holds a remade capture's file to the readers the acceptances
// use: F32 values, I64 identifiers and a BOOL mask read back as kept, by name and shape; the same tensors written again
// keep the file already there, while other values, a shape that does not hold its values, a negative identifier and a
// name kept twice are refused.
func TestCaptureTensorsAreWrittenAsTheAcceptancesReadThem(t *testing.T) {
	t.Parallel()
	fill := func(tensors *captureTensors, last float32) {
		tensors.f32("case/waveform", []float32{.5, -.25, 0, last}, 4)
		tensors.f32("case/layer_0", []float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
		tensors.ids("case/tokens", []int{3, 0, 7}, 3)
		tensors.mask("mask", []bool{true, true, false}, 3)
	}
	directory := filepath.Join(t.TempDir(), "speech-captures", "case")
	tensors := &captureTensors{}
	fill(tensors, 1)
	path := testcheck.Must(tensors.write(directory))(t)
	before := testcheck.Must(os.ReadFile(path))(t)
	source := testcheck.Must(safetensors.OpenSource(directory))(t)
	if !slices.Equal(captureF32(t, source, "case/waveform", 4), []float32{.5, -.25, 0, 1}) || len(captureF32(t, source, "case/layer_0", 1, 2, 3)) != 6 {
		t.Error("the F32 tensors read back otherwise than they were kept")
	}
	if !slices.Equal(captureIDs(t, source, "case/tokens", 3), []int{3, 0, 7}) {
		t.Error("the I64 identifiers read back otherwise than they were kept")
	}
	mask := source.Tensors["mask"]
	if data := testcheck.Must(io.ReadAll(mask.Reader()))(t); mask.DType != "BOOL" || !slices.Equal(mask.Shape, []uint64{3}) || !slices.Equal(data, []byte{1, 1, 0}) {
		t.Errorf("the mask reads back as %s %v %v, want BOOL [3] holding 1 1 0", mask.DType, mask.Shape, data)
	}
	testcheck.NoError(t, source.Close())
	again := &captureTensors{}
	fill(again, 1)
	if path, err := again.write(directory); err != nil || !slices.Equal(testcheck.Must(os.ReadFile(path))(t), before) {
		t.Errorf("the same tensors written again give %v; want the file already there kept", err)
	}
	other := &captureTensors{}
	fill(other, 2)
	if _, err := other.write(directory); err == nil {
		t.Error("other values were written over the capture already there")
	}
	for name, refused := range map[string]func(*captureTensors){
		"a shape that does not hold its values": func(c *captureTensors) { c.f32("x", []float32{1, 2}, 3) },
		"a negative identifier":                 func(c *captureTensors) { c.ids("x", []int{-1}, 1) },
		"a name kept twice":                     func(c *captureTensors) { c.f32("x", []float32{1}, 1); c.f32("x", []float32{1}, 1) },
	} {
		tensors := &captureTensors{}
		refused(tensors)
		if _, err := tensors.write(filepath.Join(t.TempDir(), "refused")); err == nil {
			t.Errorf("%s was written", name)
		}
	}
}

// TestACaptureNamesOneMaker holds both capture records to one maker a test can tell apart: overgo at a full commit
// through cmd/speech-capture with no reference runtime, or a complete Python and Torch runtime with no producer, as the
// records the reference software made stay valid. A record naming both, neither, an abbreviated or uppercase commit,
// or another command, is refused.
func TestACaptureNamesOneMaker(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	var encoder encoderCapture
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile("testdata/encoder_capture.json"))(t), &encoder))
	referenceEncoder := encoder
	referenceEncoder.Producer = captureProducer{}
	referenceEncoder.Runtime.Python, referenceEncoder.Runtime.Torch, referenceEncoder.Runtime.Transformers = election.Runtime.Python, election.Runtime.Torch, election.Runtime.Transformers
	referenceEncoder.Runtime.Device, referenceEncoder.Runtime.Attention = "cpu", "eager"
	referenceEncoder.Runtime.Files = map[string]string{"torch/functional.py": strings.Repeat("e", 64)}
	overgoEncoder := encoder
	overgoEncoder.Runtime, overgoEncoder.Producer = encoderCapture{}.Runtime, overgoMade
	digest := strings.Repeat("b", 64)
	referenceTransducer := syntheticTransducerCapture(t, election, digest)
	referenceTransducer.CaptureSHA256, referenceTransducer.TensorsSHA256, referenceTransducer.TensorCount = digest, digest, 1
	overgoTransducer := referenceTransducer
	overgoTransducer.Runtime, overgoTransducer.Producer = transducerCapture{}.Runtime, overgoMade
	makers := map[string]captureProducer{
		"an abbreviated commit": {Commit: overgoMade.Commit[:12], Command: SpeechCaptureCommand},
		"an uppercase commit":   {Commit: strings.ToUpper(overgoMade.Commit), Command: SpeechCaptureCommand},
		"another command":       {Commit: overgoMade.Commit, Command: "cmd/validate"},
	}
	encoders := map[string]func() encoderCapture{
		"both makers": func() encoderCapture { both := referenceEncoder; both.Producer = overgoMade; return both },
		"no maker":    func() encoderCapture { none := overgoEncoder; none.Producer = captureProducer{}; return none },
		"another torch": func() encoderCapture {
			other := referenceEncoder
			other.Runtime.Torch = "2.8.0+cpu"
			return other
		},
	}
	transducers := map[string]func() transducerCapture{
		"both makers": func() transducerCapture { both := referenceTransducer; both.Producer = overgoMade; return both },
		"no maker":    func() transducerCapture { none := overgoTransducer; none.Producer = captureProducer{}; return none },
	}
	for name, maker := range makers {
		encoders[name] = func() encoderCapture { made := overgoEncoder; made.Producer = maker; return made }
		transducers[name] = func() transducerCapture { made := overgoTransducer; made.Producer = maker; return made }
	}
	for name, capture := range map[string]encoderCapture{"reference": referenceEncoder, "overgo": overgoEncoder} {
		if err := capture.validate(election); err != nil {
			t.Errorf("the encoder capture %s made is refused: %v", name, err)
		}
	}
	for name, capture := range map[string]transducerCapture{"reference": referenceTransducer, "overgo": overgoTransducer} {
		if err := capture.validate(); err != nil {
			t.Errorf("the transducer capture %s made is refused: %v", name, err)
		}
	}
	for name, capture := range encoders {
		if err := capture().validate(election); err == nil {
			t.Errorf("an encoder capture with %s is admitted", name)
		}
	}
	for name, capture := range transducers {
		if err := capture().validate(); err == nil {
			t.Errorf("a transducer capture with %s is admitted", name)
		}
	}
}

// TestARemadeEncoderCaptureIsRecordedWhereTheAcceptanceReadsIt holds the remade CPU ASR reference's record to the
// acceptance's reading: the record written names overgo as its maker and the tensors file by digest and count, it
// validates against the election, its identity is the published capture's, the store finds its tensors file, and
// recording it again changes nothing.
func TestARemadeEncoderCaptureIsRecordedWhereTheAcceptanceReadsIt(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.Must(artifact.CommitBatch(t.Context(), store, testcheck.Must(election.Batch("speech-capture-test/election"))(t)))(t)
	var capture encoderCapture
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile("testdata/encoder_capture.json"))(t), &capture))
	capture.Runtime = encoderCapture{}.Runtime
	tensors := &captureTensors{}
	tensors.f32("probe", []float32{1}, 1)
	path := testcheck.Must(tensors.write(filepath.Join(t.TempDir(), "speech-captures", "asr-reference")))(t)
	recordPath := filepath.Join(t.TempDir(), "encoder_capture.json")
	id := testcheck.Must(recordEncoderCapture(t.Context(), store, election, capture, overgoMade, path, recordPath))(t)
	var recorded encoderCapture
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(recordPath))(t), &recorded))
	testcheck.NoError(t, recorded.validate(election))
	tensorID, _, count, err := captureTensorFile(path)
	testcheck.NoError(t, err)
	if recorded.Producer != overgoMade || recorded.TensorsSHA256 != tensorID.DigestHex() || recorded.TensorCount != count || count != 1 {
		t.Errorf("the record names %+v, tensors %s of %d; want overgo at its commit and the file %s of 1", recorded.Producer, recorded.TensorsSHA256, recorded.TensorCount, tensorID.DigestHex())
	}
	golden := testcheck.Must(artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-encoder-golden/v1"), recorded))(t)
	if golden.Descriptor.ID != id {
		t.Errorf("the record written reads as %s, and the store published %s", golden.Descriptor.ID, id)
	}
	testcheck.Must(artifact.RequireTypedContent(t.Context(), store, id))(t)
	testcheck.Must(artifact.AvailablePath(t.Context(), store, tensorID, artifact.LocationFile))(t)
	if again, err := recordEncoderCapture(t.Context(), store, election, capture, overgoMade, path, recordPath); err != nil || again != id {
		t.Errorf("recording the same capture again gives %s, %v; want %s", again, err, id)
	}
}

// TestARemadeTransducerCaptureTakesThePlaceOfTheOneItRemakes holds a remade transducer capture to the list the
// acceptances read: it is published with overgo as its maker, read back valid, and listed where the capture it remakes
// was, the other captures kept; a capture the list does not name is not remade into it.
func TestARemadeTransducerCaptureTakesThePlaceOfTheOneItRemakes(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.Must(artifact.CommitBatch(t.Context(), store, testcheck.Must(election.Batch("speech-capture-test/model"))(t)))(t)
	capture := syntheticTransducerCapture(t, election, strings.Repeat("b", 64))
	capture.Runtime = transducerCapture{}.Runtime
	tensors := &captureTensors{}
	tensors.f32("waveform", []float32{1}, 1)
	path := testcheck.Must(tensors.write(filepath.Join(t.TempDir(), "speech-captures", "transducer-synthetic")))(t)
	replaced := testcheck.Must(artifact.ParseID("evidence:sha256:" + strings.Repeat("c", 64)))(t)
	kept := testcheck.Must(artifact.ParseID("evidence:sha256:" + strings.Repeat("d", 64)))(t)
	listPath := filepath.Join(t.TempDir(), "transducer_captures.json")
	testcheck.NoError(t, jsonfile.Write(listPath, []artifact.ID{kept, replaced}, 0o644))
	id := testcheck.Must(recordTransducerCapture(t.Context(), store, capture, overgoMade, path, listPath, replaced))(t)
	var listed []artifact.ID
	testcheck.NoError(t, jsonfile.DecodeStrict(listPath, &listed))
	if !slices.Equal(listed, []artifact.ID{kept, id}) {
		t.Errorf("the list holds %v; want %s kept and %s in place of %s", listed, kept, id, replaced)
	}
	read := testcheck.Must(readTransducerCapture(t.Context(), store, id))(t)
	if read.Producer != overgoMade || read.Runtime.Python != "" || read.TensorCount != 1 {
		t.Errorf("the remade capture reads back made by %+v with Python %q and %d tensors; want overgo alone and 1", read.Producer, read.Runtime.Python, read.TensorCount)
	}
	if _, err := recordTransducerCapture(t.Context(), store, capture, overgoMade, path, listPath, replaced); err == nil {
		t.Error("a capture the list no longer names was remade into it")
	}
	remake := SpeechCaptureRemake{Store: store, References: t.TempDir(), Commit: overgoMade.Commit}
	if _, err := RemakeTransducerCapture(t.Context(), remake, listPath, replaced); err == nil {
		t.Error("a capture the list does not name was remade before anything was read")
	}
	remake.Commit = overgoMade.Commit[:12]
	if _, err := RemakeTransducerCapture(t.Context(), remake, listPath, id); err == nil {
		t.Error("a remake at an abbreviated commit was admitted")
	}
	if _, err := RemakeEncoderCapture(t.Context(), remake, election, filepath.Join(t.TempDir(), "encoder_capture.json")); err == nil {
		t.Error("an encoder remake at an abbreviated commit was admitted")
	}
}

// TestTheEncoderPathKeepsEveryBoundaryTheAcceptanceReads runs a small real encoder through the path a remade CPU ASR
// reference is traced with, and reads what it kept with the acceptance's own readers: the waveform, features and
// all-valid mask, the input projection and every block, the logits over the whole vocabulary and the frame
// identifiers, and nothing else. Traced again, the same tensors keep the file already written.
func TestTheEncoderPathKeepsEveryBoundaryTheAcceptanceReads(t *testing.T) {
	t.Parallel()
	const oracle = "../speechrecognition/testdata/encoder.json"
	store := overgodbtest.Open(t, t.TempDir())
	fixture := speechrecognitiontest.Publish(t, store, oracle)
	directory := testcheck.Must(artifact.AvailablePath(t.Context(), store, fixture.Definition.Model, artifact.LocationDirectory))(t)
	var captured struct {
		Declaration speechrecognition.Declaration `json:"declaration"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(oracle))(t), &captured))
	weights := testcheck.Must(safetensors.OpenSource(directory))(t)
	t.Cleanup(func() { weights.Close() })
	encoder := testcheck.Must(speechrecognition.LoadEncoder(t.Context(), weights, captured.Declaration, 1<<24))(t)
	tokenizer := testcheck.Must(hfbpe.Load(directory))(t)
	const vocabulary = 5
	path := encoderPath{frontend: tinyFrontend(t, 4), grouping: audiodsp.GroupedFeatureConfig{StackFrames: 1, FinalFrameSamples: 4},
		encoder: encoder, blocks: len(captured.Declaration.Blocks), vocabulary: vocabulary, decode: tokenizer.Decode}
	wave, samples := speechrecognitiontest.Wave(t, false)
	want := TranscriptObservation{Fixture: "synthetic", Audio: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, wave))(t), SampleRate: 16000, Samples: samples}
	written := filepath.Join(t.TempDir(), "asr-reference")
	var traced encoderCase
	for range 2 {
		tensors := &captureTensors{}
		traced = testcheck.Must(path.trace(t.Context(), tensors, want, wave))(t)
		if traced.Frames[0] != 1 || traced.Audio != want.Audio || traced.Text == "" {
			t.Fatalf("the traced case reads %+v", traced)
		}
		testcheck.Must(tensors.write(written))(t)
	}
	source := testcheck.Must(safetensors.OpenSource(written))(t)
	defer source.Close()
	frames, width := traced.Frames[1], traced.Frames[2]
	captureF32(t, source, "synthetic/waveform", int(samples))
	captureF32(t, source, "synthetic/features", frames, width)
	for _, valid := range captureIDs(t, source, "synthetic/mask", frames) {
		if valid != 1 {
			t.Fatal("the mask marks a frame of a single clip invalid")
		}
	}
	for block := -1; block < path.blocks; block++ {
		name := fmt.Sprintf("synthetic/layer_%d", block)
		if block < 0 {
			name = "synthetic/input_projection"
		}
		if tensor, found := source.Tensors[name]; !found || tensor.DType != "F32" || len(tensor.Shape) != 2 {
			t.Errorf("the boundary %s is absent or not an F32 matrix", name)
		}
	}
	tokens, found := source.Tensors["synthetic/tokens"]
	if !found || len(tokens.Shape) != 1 {
		t.Fatal("the frame identifiers are absent")
	}
	outFrames := int(tokens.Shape[0])
	captureF32(t, source, "synthetic/logits", outFrames, vocabulary)
	captureIDs(t, source, "synthetic/tokens", outFrames)
	if len(source.Tensors) != 6+path.blocks {
		t.Errorf("the capture keeps %d tensors; want the %d the acceptance reads", len(source.Tensors), 6+path.blocks)
	}
}

// TestTheTransducerPathKeepsEveryBoundaryTheAcceptancesRead runs a tiny transducer through the path a remade transducer
// capture is traced with, and reads what it kept by the names the acceptances read: the waveform, the BOOL valid-frame
// mask and features, each mode's encoder boundaries per chunk, offline decoder steps from zero and streaming ones from
// one, as the reference's initialization forward took the first, and each mode's emissions; its traces name every
// chunk it was fed and every step's token, and nothing else is kept.
func TestTheTransducerPathKeepsEveryBoundaryTheAcceptancesRead(t *testing.T) {
	t.Parallel()
	values, shapes := map[string][]float32{}, map[string][]int{}
	add := func(name string, shape []int, data []float32) string {
		values[name], shapes[name] = data, shape
		return name
	}
	identity := speechrecognition.AffineBinding{Weight: add("identity", []int{2, 2}, []float32{1, 0, 0, 1})}
	normalize := speechrecognition.AffineBinding{Weight: add("norm.weight", []int{2}, []float32{1, 1}), Bias: add("norm.bias", []int{2}, []float32{0, 0})}
	feedForward := speechrecognition.FeedForwardBinding{Norm: normalize, In: identity, Out: identity}
	relative := &speechrecognition.ProjectedRelativeBinding{Projection: identity, ContentBias: add("content.bias", []int{1, 2}, []float32{0, 0}),
		PositionBias: add("position.bias", []int{1, 2}, []float32{0, 0}), Base: 10000, LeftContext: 2, MaxPositions: 256}
	block := speechrecognition.BlockBinding{First: feedForward, Second: feedForward, OutNorm: normalize,
		Attention: speechrecognition.AttentionBinding{Norm: normalize, Query: identity, Key: identity, Value: identity, Out: identity, Heads: 1, BlockFrames: 2, ProjectedRelative: relative},
		Convolution: speechrecognition.ConvolutionBinding{Norm: normalize, In: speechrecognition.AffineBinding{Weight: add("glu", []int{4, 2}, []float32{1, 0, 0, 1, 0, 0, 0, 0})},
			Kernel: add("depthwise", []int{2, 1, 3}, []float32{0, 0, 1, 0, 0, 1}), LayerNorm: normalize, Out: identity, Stride: 1, Causal: true}}
	declaration := speechrecognition.Declaration{Input: identity, Output: identity, Blocks: []speechrecognition.BlockBinding{block}, Activation: "silu", FeedForwardScale: .5, LayerNormEpsilon: 1e-5, BatchNormEpsilon: 1e-5}
	binding := speechrecognition.TransducerBinding{Bands: 2,
		Subsampling: []speechrecognition.SpatialConvolutionBinding{{Affine: speechrecognition.AffineBinding{Weight: add("spatial", []int{1, 1, 1, 1}, []float32{1})}, Stride: [2]uint32{1, 1}}},
		ConditionIn: speechrecognition.AffineBinding{Weight: add("condition", []int{2, 3}, []float32{1, 0, 0, 0, 1, 0})}, ConditionOut: identity, ConditionSlots: 1,
		Embedding: add("embedding", []int{3, 2}, []float32{.1, .2, .3, .4, .5, .6}),
		Recurrent: []speechrecognition.RecurrentBinding{{Input: speechrecognition.AffineBinding{Weight: add("lstm.input", []int{8, 2}, slices.Repeat([]float32{.5}, 16))},
			Recurrent: speechrecognition.AffineBinding{Weight: add("lstm.recurrent", []int{8, 2}, make([]float32, 16))}}},
		DecoderProjection: identity, Joint: speechrecognition.AffineBinding{Weight: add("joint", []int{3, 2}, make([]float32, 6)), Bias: add("joint.bias", []int{3}, []float32{0, 1, -1})},
		Blank: 2, MaxSymbolsPerFrame: 2}
	directory := t.TempDir()
	testcheck.NoError(t, safetensors.Save(filepath.Join(directory, "model.safetensors"), values, shapes, nil))
	weights := testcheck.Must(safetensors.OpenSource(directory))(t)
	t.Cleanup(func() { weights.Close() })
	model := testcheck.Must(speechrecognition.LoadTransducer(t.Context(), weights, declaration, binding, 1<<24))(t)
	path := transducerPath{hop: 2, bands: binding.Bands, blocks: len(declaration.Blocks), recurrent: len(binding.Recurrent), model: model,
		decode: func(ids []int) (string, error) { return fmt.Sprint(ids), nil }}
	pcm := make([]float32, 64)
	for index := range pcm {
		pcm[index] = float32(index%8) / 8
	}
	tensors := &captureTensors{}
	traces := testcheck.Must(path.trace(t.Context(), tensors, tinyFrontend(t, 2), pcm, transducerCapture{SampleRate: 16000, FirstChunkFrames: 2, FollowingChunkFrames: 2}))(t)
	written := filepath.Join(t.TempDir(), "transducer-synthetic")
	testcheck.Must(tensors.write(written))(t)
	source := testcheck.Must(safetensors.OpenSource(written))(t)
	defer source.Close()
	features := source.Tensors["features"]
	frames := int(features.Shape[0])
	captureF32(t, source, "waveform", len(pcm))
	captureF32(t, source, "features", frames, binding.Bands)
	if mask := source.Tensors["mask"]; mask.DType != "BOOL" || !slices.Equal(mask.Shape, []uint64{uint64(frames)}) {
		t.Errorf("the mask is %s %v; want BOOL [%d]", mask.DType, mask.Shape, frames)
	}
	count := 3
	for mode, first := range map[string]int{"offline": 0, "streaming": 1} {
		trace := traces[mode]
		if len(trace.Chunks) == 0 || len(trace.Steps) == 0 || trace.Text == "" {
			t.Fatalf("the %s trace is incomplete: %+v", mode, trace)
		}
		for chunk := range trace.Chunks {
			for boundary := -1; boundary < path.blocks; boundary++ {
				name := fmt.Sprintf("%s/chunk_%d/layer_%d", mode, chunk, boundary)
				if boundary < 0 {
					name = fmt.Sprintf("%s/chunk_%d/subsampling", mode, chunk)
				}
				if tensor, found := source.Tensors[name]; !found || len(tensor.Shape) != 3 || tensor.Shape[0] != 1 {
					t.Errorf("the encoder boundary %s is absent or not [1 frames width]", name)
				}
			}
		}
		for step := range trace.Steps {
			key := fmt.Sprintf("%s/step_%d/", mode, step+first)
			captureF32(t, source, key+"decoder", 1, 1, 2)
			captureF32(t, source, key+"hidden", 1, 1, 2)
			captureF32(t, source, key+"cell", 1, 1, 2)
			captureF32(t, source, key+"logits", 1, 1, 1, 3)
		}
		for _, name := range []string{"sequences", "durations"} {
			tensor := source.Tensors[mode+"/"+name]
			if len(tensor.Shape) != 2 || tensor.Shape[0] != 1 {
				t.Fatalf("the %s %s are absent or not [1 n]", mode, name)
			}
			captureIDs(t, source, mode+"/"+name, 1, int(tensor.Shape[1]))
		}
		count += len(trace.Chunks)*(path.blocks+1) + 4*len(trace.Steps) + 2
	}
	if _, found := source.Tensors["streaming/step_0/logits"]; found {
		t.Error("a streaming step is kept as the initialization forward the acceptance skips")
	}
	if chunks := len(traces["streaming"].Chunks); chunks != (frames+1)/2 || traces["offline"].Chunks[0].Frames != frames {
		t.Errorf("the traces name %d streaming chunks and an offline chunk of %d frames; want %d and %d", chunks, traces["offline"].Chunks[0].Frames, (frames+1)/2, frames)
	}
	if len(source.Tensors) != count {
		t.Errorf("the capture keeps %d tensors; want the %d the acceptances read", len(source.Tensors), count)
	}
}
