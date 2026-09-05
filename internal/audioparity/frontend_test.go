package audioparity

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/binaryschema"
	"overgo/internal/media"
	"overgo/internal/testutil"
)

type frontendTensor struct {
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
	Data   []byte `json:"data"`
}

func (tensor frontendTensor) values(t *testing.T, shape ...int) []float32 {
	t.Helper()
	if !slices.Equal(tensor.Shape, shape) {
		t.Fatalf("tensor shape %v, want %v", tensor.Shape, shape)
	}
	count := 1
	for _, dimension := range shape {
		if dimension <= 0 || dimension > len(tensor.Data)/count {
			t.Fatal("invalid tensor extent")
		}
		count *= dimension
	}
	if count != len(tensor.Data)/4 || len(tensor.Data)%4 != 0 {
		t.Fatal("tensor byte extent differs from float32 shape")
	}
	digest := sha256.Sum256(tensor.Data)
	if hex.EncodeToString(digest[:]) != tensor.SHA256 {
		t.Fatal("tensor payload hash differs")
	}
	values := make([]float32, count)
	for index := range values {
		values[index] = binaryschema.LittleEndian.Float32(tensor.Data[index*binaryschema.Uint32Bytes:])
		if math.IsNaN(float64(values[index])) || math.IsInf(float64(values[index]), 0) {
			t.Fatal("non-finite golden tensor")
		}
	}
	return values
}

func TestFrontendIntermediateParity(t *testing.T) {
	var fixture struct {
		Version               int            `json:"version"`
		ResampleFixtureSHA256 string         `json:"resample_fixture_sha256"`
		Election              artifact.ID    `json:"election"`
		Source                SourceIdentity `json:"source"`
		Dataset               DatasetBinding `json:"dataset"`
		Fixture               string         `json:"fixture"`
		EncodedSHA256         string         `json:"encoded_sha256"`
		ConfigSHA256          string         `json:"config_sha256"`
		Runtime               struct {
			Python, Torch, Transformers, Device string
			Files                               map[string]string
		} `json:"runtime"`
		Cases []struct {
			Name           string                    `json:"name"`
			SampleRate     int                       `json:"sample_rate"`
			FrameLimit     int                       `json:"frame_limit"`
			PaddingSamples int                       `json:"padding_samples"`
			Tensors        map[string]frontendTensor `json:"tensors"`
		} `json:"cases"`
	}
	encoded, err := os.ReadFile("testdata/frontend_boundaries.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &fixture); err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("testdata/granite_speech_5_election.json")
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.Version != 1 || fixture.Election != election.ID || fixture.Source != election.ModelSource || !reflect.DeepEqual(fixture.Dataset, election.Dataset) ||
		fixture.Fixture != election.Observations[0].Fixture || fixture.EncodedSHA256 != election.Observations[0].Audio.DigestHex() ||
		fixture.Runtime.Python != election.Runtime.Python || fixture.Runtime.Torch != election.Runtime.Torch ||
		fixture.Runtime.Transformers != election.Runtime.Transformers || fixture.Runtime.Device != "cpu" || len(fixture.Runtime.Files) == 0 || len(fixture.Cases) == 0 {
		t.Fatal("frontend capture does not bind the elected CPU oracle")
	}
	encoded, err = os.ReadFile("../audiodsp/testdata/power_logmel_frontend.json")
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(bytes.ReplaceAll(encoded, []byte("\r\n"), []byte("\n")))
	if hex.EncodeToString(digest[:]) != fixture.ConfigSHA256 {
		t.Fatal("frontend declaration hash differs")
	}
	var config audiodsp.FrontendConfig
	if err := json.Unmarshal(encoded, &config); err != nil {
		t.Fatal(err)
	}
	t.Run("resample", func(t *testing.T) {
		encoded, err := os.ReadFile(testutil.FixturePath(t, "pockettts", "g9_resample.json"))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(bytes.ReplaceAll(encoded, []byte("\r\n"), []byte("\n")))
		if hex.EncodeToString(digest[:]) != fixture.ResampleFixtureSHA256 {
			t.Fatal("resampling oracle fixture changed")
		}
		var golden struct {
			Up, Down   int
			Taps, X, Y []float64
		}
		if err := json.Unmarshal(encoded, &golden); err != nil {
			t.Fatal(err)
		}
		if golden.Up <= 0 || golden.Down <= 0 || len(golden.X) == 0 || len(golden.Y) == 0 {
			t.Fatal("empty resampling oracle")
		}
		input := make([]float32, len(golden.X))
		var maximum, tapSum, outputMaximum float64
		for index, value := range golden.X {
			input[index] = float32(value)
			maximum = max(maximum, math.Abs(value))
		}
		for _, tap := range golden.Taps {
			tapSum += math.Abs(tap)
		}
		for _, value := range golden.Y {
			outputMaximum = max(outputMaximum, math.Abs(value))
		}
		// Absolute input-quantization propagation through the FIR plus output
		// float32 rounding; no distributional accumulation assumption.
		tolerance := float64(math.Nextafter32(1, 2)-1) * (maximum*tapSum + outputMaximum)
		resampled, err := media.ResamplePoly(t.Context(), nil, input, golden.Up, golden.Down, golden.Taps)
		if err != nil {
			t.Fatal(err)
		}
		testutil.RequireWithin(t, "resampled-waveform", resampled, golden.Y, tolerance)
		declaration := config
		declaration.SampleRate *= golden.Up
		declaration.ResampleTaps = golden.Taps
		plan, err := audiodsp.NewFrontend(declaration, uint64(math.MaxInt))
		if err != nil {
			t.Fatal(err)
		}
		var workspace audiodsp.Workspace
		for _, chunks := range [][][]float32{{input}, {input[:1], nil, input[1:]}} {
			reconstructed, err := plan.Reconstruct(t.Context(), chunks, config.SampleRate*golden.Down, &workspace)
			if err != nil {
				t.Fatal(err)
			}
			testutil.RequireWithin(t, "resampled-STFT-roundtrip", reconstructed, golden.Y, tolerance)
		}
	})
	plan, err := audiodsp.NewFrontend(config, uint64(math.MaxInt))
	if err != nil {
		t.Fatal(err)
	}
	for _, probe := range fixture.Cases {
		t.Run(probe.Name, func(t *testing.T) {
			input := probe.Tensors["input"].values(t, probe.Tensors["input"].Shape[0])
			waveform := append(slices.Clone(input), make([]float32, probe.PaddingSamples)...)
			testutil.RequireWithin(t, "waveform", waveform, probe.Tensors["waveform"].values(t, len(waveform)), 0)
			frames, bands, bins := probe.FrameLimit, int(config.Geometry.FeatureBins), config.FFTLength/2+1
			for _, chunkSize := range []int{len(waveform), int(config.Geometry.HopSamples), 1} {
				var chunks [][]float32
				for start := 0; start < len(waveform); start += chunkSize {
					chunks = append(chunks, waveform[start:min(start+chunkSize, len(waveform))])
				}
				observed := map[string][]float64{}
				var logs []float32
				var seen int
				options := audiodsp.ProcessOptions{FrameLimit: frames, Observe: func(trace audiodsp.FrameTrace) error {
					if trace.Frame != seen {
						t.Fatalf("frame %d, want %d", trace.Frame, seen)
					}
					seen++
					for name, values := range map[string][]float64{"window": trace.Window, "real": trace.Real, "imaginary": trace.Imaginary, "energy": trace.Energy, "mel": trace.Mel} {
						observed[name] = append(observed[name], values...)
					}
					logs = append(logs, trace.LogMel...)
					return nil
				}}
				var workspace audiodsp.Workspace
				normalized, gotFrames, err := plan.Process(t.Context(), chunks, probe.SampleRate, &workspace, options)
				if err != nil {
					t.Fatal(err)
				}
				if gotFrames != frames || seen != frames {
					t.Fatalf("frames %d observed %d, want %d", gotFrames, seen, frames)
				}
				// The declared protocol allows one float32 FFT-length error budget,
				// scaled to each stage's reference magnitude. This is a comparison
				// budget, not a formal bound for logarithmic conditioning.
				tolerance := func(values []float32) float64 {
					maximum := 1.0
					for _, value := range values {
						maximum = max(maximum, math.Abs(float64(value)))
					}
					return float64(math.Nextafter32(1, 2)-1) * float64(config.FFTLength) * maximum
				}
				for name, width := range map[string]int{"window": int(config.Geometry.WindowSamples), "real": bins, "imaginary": bins, "energy": bins, "mel": bands} {
					want := probe.Tensors[name].values(t, frames, width)
					testutil.RequireWithin(t, name, observed[name], want, tolerance(want))
				}
				for name, got := range map[string][]float32{"log_mel": logs, "normalized": normalized} {
					want := probe.Tensors[name].values(t, frames, bands)
					testutil.RequireWithin(t, name, got, want, tolerance(want))
				}
			}
		})
	}
}
