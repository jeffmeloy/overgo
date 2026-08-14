//go:build integration

package adaptiveparity

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const (
	pocketPCMMaxDiff = 2e-2
	pocketPCMRMSDiff = 1e-3
	pocketWarmLimit  = 700 * time.Millisecond
	pocketHeapLimit  = 600 << 20
)

type pocketTensor struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
}

type pocketBackboneGolden struct {
	Text             string `json:"text"`
	TransformerCalls []struct {
		In pocketTensor `json:"in"`
	} `json:"transformer_calls"`
}

type pocketGenerationGolden struct {
	FlowCalls []struct {
		Noise pocketTensor `json:"noise"`
		Out   pocketTensor `json:"out"`
	} `json:"flow_calls"`
	NQuantizerCalls int          `json:"n_quantizer_calls"`
	EOSLogits       []float64    `json:"eos_logits"`
	PCM             pocketTensor `json:"pcm"`
}

func TestPocketTTSProductionParity(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	directory := pocketArtifactDirectory(t)

	t.Run("compiled-recipe", func(t *testing.T) {
		synthesizer, err := speechsynth.LoadSynthesizer(directory)
		if err != nil {
			t.Fatalf("UNAVAILABLE: Pocket-TTS artifact cannot load; parity NOT verified: %v", err)
		}
		capability := modelrecipetest.NewCapability(t, "pocket-tts", recipe.TaskSpeech)
		if err := speechsynth.RegisterRuntime(capability.Runtime, capability.Model, synthesizer); err != nil {
			t.Fatal(err)
		}
		audio := modelrecipetest.MustExecuteScalar[speechsynth.Audio](t, capability, "pocket-tts/production", speechsynth.SynthesisRequest{
			Text: "Green always green.", MaxFrames: 2, Seed: 7,
		})
		requireMonoAudio(t, audio)
	})

	t.Run("reference-waveform", func(t *testing.T) {
		runtime.GC()
		var baseline runtime.MemStats
		runtime.ReadMemStats(&baseline)
		var peak atomic.Uint64
		peak.Store(baseline.HeapAlloc)
		done := make(chan struct{})
		stopped := make(chan struct{})
		go samplePocketHeap(done, stopped, &peak)
		sampling := true
		stopSampling := func() {
			if sampling {
				close(done)
				<-stopped
				sampling = false
			}
		}
		defer stopSampling()

		model, coldWall := loadPocketModel(t, directory)
		backbone := loadPocketBackboneFixture(t)
		generation := loadPocketFixture[pocketGenerationGolden](t, "g7_generation.json")
		tokens := loadPocketFixture[struct {
			IDs []int `json:"ids"`
		}](t, "g1_tokenizer.json")

		if len(backbone.TransformerCalls) != 1 || len(backbone.TransformerCalls[0].In.Shape) != 3 ||
			generation.NQuantizerCalls <= 0 || len(generation.FlowCalls) < generation.NQuantizerCalls+2 ||
			len(generation.EOSLogits) < generation.NQuantizerCalls+2 {
			t.Fatal("Pocket-TTS oracle geometry is incomplete")
		}
		voice := pocketF32(backbone.TransformerCalls[0].In.Values)
		voiceFrames := backbone.TransformerCalls[0].In.Shape[1]
		noises := make([][]float32, generation.NQuantizerCalls)
		for step := range noises {
			noises[step] = pocketF32(generation.FlowCalls[2+step].Noise.Values)
		}

		walls := make([]time.Duration, 3)
		var pcm []float32
		for run := range walls {
			started := time.Now()
			latents, eos, err := model.GenerateLatents(voice, voiceFrames, tokens.IDs, speechsynth.GenerateParams{
				MaxFrames: generation.NQuantizerCalls, EOSThreshold: math.Inf(1),
				NoiseAt: func(step int, destination []float32) { copy(destination, noises[step]) },
			})
			if err != nil {
				t.Fatal(err)
			}
			for frame := range latents.Frames {
				want := generation.FlowCalls[2+frame]
				row := latents.Values[frame*latents.Width : (frame+1)*latents.Width]
				if len(want.Noise.Values) != len(row) || len(want.Out.Values) != len(row) {
					t.Fatalf("latent frame %d oracle width mismatch", frame)
				}
				for index := range row {
					reference := float32(want.Noise.Values[index] + want.Out.Values[index])
					if math.Abs(float64(row[index]-reference)) > 1e-2 {
						t.Fatalf("latent frame %d value %d diverges", frame, index)
					}
				}
				if math.Abs(eos[frame]-generation.EOSLogits[2+frame]) > 5e-2 {
					t.Fatalf("EOS frame %d diverges", frame)
				}
			}
			pcm, err = model.LatentsToPCM(latents)
			if err != nil {
				t.Fatal(err)
			}
			walls[run] = time.Since(started)
		}
		sort.Slice(walls, func(left, right int) bool { return walls[left] < walls[right] })
		if os.Getenv("OVERGO_POCKET_TTS_BASELINE") == "1" && walls[1] > pocketWarmLimit {
			t.Fatalf("warm synthesis %s exceeds %s", walls[1], pocketWarmLimit)
		}

		maxDiff, rms, wantRMS := pocketWaveformDiff(pcm, generation.PCM.Values)
		if maxDiff > pocketPCMMaxDiff || math.Abs(rms-wantRMS) > pocketPCMRMSDiff {
			t.Fatalf("waveform max=%.6g rms=%.6g reference_rms=%.6g", maxDiff, rms, wantRMS)
		}
		encoded, err := media.EncodeWAVPCM16(pcm, model.Codec.SampleRate)
		if err != nil {
			t.Fatal(err)
		}
		decoded, rate, err := media.DecodeWAV(encoded)
		if err != nil {
			t.Fatal(err)
		}
		requireMonoAudio(t, speechsynth.Audio{PCM: decoded, SampleRate: rate, Channels: 1})
		if len(decoded) != len(pcm) {
			t.Fatalf("WAV samples = %d, want %d", len(decoded), len(pcm))
		}
		stopSampling()
		var final runtime.MemStats
		runtime.ReadMemStats(&final)
		updatePocketPeak(&peak, final.HeapAlloc)
		peakBytes := peak.Load() - baseline.HeapAlloc
		if peakBytes > pocketHeapLimit {
			t.Fatalf("peak heap %d exceeds %d", peakBytes, pocketHeapLimit)
		}
		t.Logf("Pocket-TTS production parity: cold load %s; warm synth median %s; peak heap %.3fGiB; %d samples @ %dHz mono; max %.6g; rms %.6g",
			coldWall, walls[1], float64(peakBytes)/(1<<30), len(pcm), rate, maxDiff, rms)
	})
}

func pocketArtifactDirectory(t *testing.T) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(roots.Models, "pocket-tts")
	if _, err := os.Stat(filepath.Join(directory, "pockettts_config.json")); err != nil {
		t.Fatalf("UNAVAILABLE: Pocket-TTS artifact absent; parity NOT verified: %v", err)
	}
	return directory
}

func loadPocketModel(t *testing.T, directory string) (*speechsynth.Model, time.Duration) {
	t.Helper()
	started := time.Now()
	model, err := speechsynth.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return model, time.Since(started)
}

func loadPocketFixture[Value any](t *testing.T, name string) Value {
	t.Helper()
	var value Value
	raw, err := os.ReadFile(testutil.FixturePath(t, "pockettts", name))
	if err != nil {
		t.Fatalf("UNAVAILABLE: Pocket-TTS %s absent; parity NOT verified: %v", name, err)
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func loadPocketBackboneFixture(t *testing.T) pocketBackboneGolden {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "pockettts", "g6_backbone.json"))
	if err != nil {
		t.Fatalf("UNAVAILABLE: Pocket-TTS g6_backbone.json absent; parity NOT verified: %v", err)
	}
	var envelope struct {
		Text             string            `json:"text"`
		TransformerCalls []json.RawMessage `json:"transformer_calls"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatal(err)
	}
	if len(envelope.TransformerCalls) == 0 {
		t.Fatal("Pocket-TTS backbone fixture has no transformer calls")
	}
	var first struct {
		In pocketTensor `json:"in"`
	}
	if err := json.Unmarshal(envelope.TransformerCalls[0], &first); err != nil {
		t.Fatal(err)
	}
	return pocketBackboneGolden{Text: envelope.Text, TransformerCalls: []struct {
		In pocketTensor `json:"in"`
	}{{In: first.In}}}
}

func pocketF32(values []float64) []float32 {
	converted := make([]float32, len(values))
	for index, value := range values {
		converted[index] = float32(value)
	}
	return converted
}

func pocketWaveformDiff(got []float32, want []float64) (float64, float64, float64) {
	if len(got) != len(want) || len(got) == 0 {
		return math.Inf(1), 0, 0
	}
	var maxDiff, gotEnergy, wantEnergy float64
	for index, value := range got {
		difference := math.Abs(float64(value) - want[index])
		maxDiff = max(maxDiff, difference)
		gotEnergy += float64(value) * float64(value)
		wantEnergy += want[index] * want[index]
	}
	return maxDiff, math.Sqrt(gotEnergy / float64(len(got))), math.Sqrt(wantEnergy / float64(len(want)))
}

func requireMonoAudio(t *testing.T, audio speechsynth.Audio) {
	t.Helper()
	if audio.SampleRate != 24000 || audio.Channels != 1 || len(audio.PCM) == 0 {
		t.Fatalf("audio contract = %dHz/%dch/%d samples", audio.SampleRate, audio.Channels, len(audio.PCM))
	}
	for index, sample := range audio.PCM {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatalf("audio sample %d is not finite", index)
		}
	}
}

func samplePocketHeap(done <-chan struct{}, stopped chan<- struct{}, peak *atomic.Uint64) {
	defer close(stopped)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			var sample runtime.MemStats
			runtime.ReadMemStats(&sample)
			updatePocketPeak(peak, sample.HeapAlloc)
		}
	}
}

func updatePocketPeak(peak *atomic.Uint64, value uint64) {
	for prior := peak.Load(); value > prior && !peak.CompareAndSwap(prior, value); prior = peak.Load() {
	}
}
