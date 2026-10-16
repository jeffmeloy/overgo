//go:build modeltest

package speechsynth

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// servedSpeechScript writes the fixture this test reads, from the model's own
// code. It is kept where Python is kept, on the machine that ran it and not
// in git: the fixture is the data.
const servedSpeechScript = "scripts/python_golden_references/pockettts_served_speech.py"

// packedFloats reads a fixture's base64 of little-endian float32 values.
func packedFloats(t *testing.T, packed string) []float32 {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(packed)
	if err != nil || len(raw)%4 != 0 {
		t.Fatalf("the fixture's packed values are unreadable: %d bytes, %v", len(raw), err)
	}
	values := make([]float32, len(raw)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return values
}

// TestServedSpeechMatchesUpstream holds what a speech request is served, from
// its text to the samples of its recording, to the model's own code speaking
// the same text in the same stored voice: the request is given the noise that
// code drew for each frame, and must then end its speech in the same frame
// and give the same samples.
//
// It is the stretch no other test reaches. The stage tests hold the model's
// arithmetic to that code with the voice handed in as conditioning rows, for
// one text of two words, stopped at a frame count the test names. Here the
// voice is the state the model directory stores, read as a served request
// reads it; the text is prepared, split and budgeted as a served request's is;
// the speech stops where the model itself says it has ended, and runs on for
// the frames after that the model's own code runs on for. Four of the texts
// are the round-trip evaluation's, and two are short ones, which are spoken on
// for longer.
//
// What it does not reach is the drawing of the noise: a served request draws
// normal values from its seed with Go's generator, and that code with torch's.
func TestServedSpeechMatchesUpstream(t *testing.T) {
	t.Run("English preset and prepared voice", func(t *testing.T) {
		testServedSpeech(t, "pockettts_served_speech.json", artifactDir(t), true)
	})
	t.Run("German preset", func(t *testing.T) {
		testServedSpeech(t, "pockettts_german_speech.json", filepath.Join(artifactDir(t), "languages", "german"), false)
	})
}

func testServedSpeech(t *testing.T, fixtureName, directory string, prepareVoice bool) {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, fixtureName))
	if err != nil {
		t.Fatalf("the upstream recordings are absent (%v); write them with %s", err, servedSpeechScript)
	}
	var fixture struct {
		Voice              string  `json:"voice"`
		Seed               int64   `json:"seed"`
		Temperature        float64 `json:"temperature"`
		EndThreshold       float64 `json:"end_threshold"`
		SampleRate         int     `json:"sample_rate"`
		FrameSamples       int     `json:"frame_samples"`
		NoiseWidth         int     `json:"noise_width"`
		WeightsSHA256      string  `json:"weights_sha256"`
		TokenizerSHA256    string  `json:"tokenizer_sha256"`
		VoiceSHA256        string  `json:"voice_sha256"`
		ReferenceRecording string  `json:"reference_recording"`
		ReferenceState     []struct {
			Frames     int
			Key, Value string
		} `json:"reference_state"`
		Cases []struct {
			Text            string  `json:"text"`
			Frames          int     `json:"frames"`
			EndSeenAt       int     `json:"end_seen_at"`
			NearestEndLogit float64 `json:"nearest_end_logit"`
			Noise           string  `json:"noise"`
			Samples         string  `json:"samples"`
			ReferenceVoice  bool    `json:"reference_voice"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture holds no text")
	}
	synth, err := LoadSynthesizer(directory)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := Inventory(synth.directory)
	if err != nil {
		t.Fatal(err)
	}
	weightMatch := false
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeights && component.Artifact.DigestHex() == fixture.WeightsSHA256 {
			weightMatch = true
		}
	}
	if !weightMatch {
		t.Fatal("independent voice fixture weights differ from served inventory")
	}
	voicePath, err := ResolveVoicePath(directory, fixture.Voice)
	if err != nil {
		t.Fatal(err)
	}
	for path, expected := range map[string]string{filepath.Join(directory, "tokenizer.model"): fixture.TokenizerSHA256, voicePath: fixture.VoiceSHA256} {
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil || id.DigestHex() != expected {
			t.Fatalf("independent input %s differs: %v", path, err)
		}
	}
	var preparedID artifact.ID
	if prepareVoice {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		batch, err := inventory.Batch("voice-reference-model")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, inventory.Manifest.ID)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "voice-reference-recipe", definition); err != nil {
			t.Fatal(err)
		}
		if err := synth.RestoreRecipe(t.Context(), store, definition); err != nil {
			t.Fatal(err)
		}
		recording, err := base64.StdEncoding.DecodeString(fixture.ReferenceRecording)
		if err != nil || len(recording) == 0 {
			t.Fatalf("independent recording absent: %v", err)
		}
		content, err := (artifact.DocumentContract{Kind: artifact.KindFile, MediaType: media.WAVMediaType, Schema: "fixture/reference-voice"}).ContentBytes(recording)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(t.Context(), store, artifact.Batch{Key: "voice-reference-recording", Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		decoded, _, err := media.DecodeAudio(t.Context(), recording, uint64(len(recording)))
		if err != nil {
			t.Fatal(err)
		}
		// Two complete model payloads bound this fixture's additional preparation tensors.
		var voiceBudget int64
		for _, component := range inventory.Components {
			voiceBudget += 2 * int64(component.Size)
		}
		preparedID, err = synth.BindVoice(t.Context(), store, definition, SynthesisRequest{ReferenceAudio: content.Descriptor.ID, ReferenceEndSample: len(decoded.Samples), VoiceMemoryBytes: voiceBudget})
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := synth.voiceState(preparedID.String())
		if err != nil || len(fixture.ReferenceState) != synth.model.Dims.Layers {
			t.Fatalf("independent state absent: %v", err)
		}
		for layer, state := range fixture.ReferenceState {
			if prepared.Frames != state.Frames {
				t.Fatalf("reference voice offset differs at layer %d", layer)
			}
			for index, want := range [][]float32{packedFloats(t, state.Key), packedFloats(t, state.Value)} {
				got := [][]float32{prepared.K[layer], prepared.V[layer]}[index]
				if len(got) != len(want) {
					t.Fatal("reference voice state geometry differs")
				}
				for j := range want {
					if math.Abs(float64(got[j]-want[j])) > tolTransformer {
						t.Fatalf("voice layer %d state %d sample %d differs by %g", layer, index, j, got[j]-want[j])
					}
				}
			}
		}
	}
	if fixture.Temperature != synth.defaultTemperature || fixture.EndThreshold != referenceEOSThreshold ||
		fixture.SampleRate != synth.model.Codec.SampleRate || fixture.NoiseWidth != synth.model.Dims.LatentDim {
		t.Fatalf("the model's own code spoke at temperature %g, ending above %g, at %d Hz from noise of %d values, and a served request speaks at %g, %g, %d and %d",
			fixture.Temperature, fixture.EndThreshold, fixture.SampleRate, fixture.NoiseWidth,
			synth.defaultTemperature, referenceEOSThreshold, synth.model.Codec.SampleRate, synth.model.Dims.LatentDim)
	}
	short, long, preparedCases, worst := 0, 0, 0, 0.0
	for _, reference := range fixture.Cases {
		noise, samples := packedFloats(t, reference.Noise), packedFloats(t, reference.Samples)
		if len(noise) != (reference.Frames+1)*fixture.NoiseWidth || len(samples) != reference.Frames*fixture.FrameSamples {
			t.Fatalf("%q: the fixture holds %d noise values and %d samples for %d frames", reference.Text, len(noise), len(samples), reference.Frames)
		}
		var plans []generationPlan
		beyond := false
		request := DocumentRequest{Text: reference.Text, Voice: fixture.Voice, Seed: fixture.Seed}
		if reference.ReferenceVoice {
			request.Voice, request.PreparedVoice = "", preparedID
			preparedCases++
		}
		audio, err := synth.speakFrom(t.Context(), request, func(plan generationPlan) frameNoise {
			plans = append(plans, plan)
			return func(step int, values []float32) {
				// The noise of the frame that code stopped at is the last it drew: a
				// request that asks for more has spoken past where that code ended.
				if step >= reference.Frames {
					beyond = true
					clear(values)
					return
				}
				copy(values, noise[step*fixture.NoiseWidth:(step+1)*fixture.NoiseWidth])
			}
		})
		if err != nil {
			t.Fatalf("%q: %v", reference.Text, err)
		}
		if len(plans) != 1 {
			t.Fatalf("%q is served in %d segments, and the model's own code speaks it as one", reference.Text, len(plans))
		}
		if after := reference.Frames - reference.EndSeenAt; plans[0].framesAfterEnd != after {
			t.Errorf("%q is spoken on for %d frames after its end, and the model's own code for %d", reference.Text, plans[0].framesAfterEnd, after)
		}
		if beyond || !audio.Complete || len(audio.PCM) != len(samples) {
			t.Errorf("%q is served as %d samples (ended on its own: %v), and the model's own code speaks %d frames of %d, the end first seen in frame %d; its logit nearest the threshold of %g is %g",
				reference.Text, len(audio.PCM), audio.Complete, reference.Frames, fixture.FrameSamples, reference.EndSeenAt, fixture.EndThreshold, reference.NearestEndLogit)
			continue
		}
		difference := 0.0
		for index, sample := range audio.PCM {
			difference = max(difference, math.Abs(float64(sample)-float64(samples[index])))
		}
		t.Logf("%q: %d frames, the end seen in frame %d, the largest difference of a sample %.3g", reference.Text, reference.Frames, reference.EndSeenAt, difference)
		if difference > tolE2EPCM {
			t.Errorf("%q: a served sample is %g from the model's own code's, beyond %g", reference.Text, difference, tolE2EPCM)
		}
		worst = max(worst, difference)
		if plans[0].framesAfterEnd == shortTextGuess+framesBeyondGuess {
			short++
		} else {
			long++
		}
	}
	if prepareVoice && preparedCases == 0 {
		t.Error("independent prepared-voice speech case is absent")
	}
	if !t.Failed() && (short == 0 || long == 0) {
		t.Fatalf("the fixture holds %d short texts and %d longer ones, and both rules need a text", short, long)
	}
	t.Logf("%d texts, %d of them short: a served request ends each in the frame the model's own code ends it in and gives its samples within %.3g", len(fixture.Cases), short, worst)
}
