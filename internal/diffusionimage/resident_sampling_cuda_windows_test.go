//go:build windows

package diffusionimage

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/processmeasure"
	"overgo/internal/testevidence"
)

// samplingHostBoundary retains the pre-residency sampling contract as an A/B
// reference: the same CUDA forward, followed by the original host Euler update.
func samplingHostBoundary(ctx context.Context, forward *ResidentForward, steps int, seed int64) ([]float32, error) {
	rng := rand.New(rand.NewSource(seed))
	pixels := make([]float32, forward.image.channels*forward.image.height*forward.image.width)
	for index := range pixels {
		pixels[index] = float32(rng.NormFloat64())
	}
	delta := float32(1) / float32(steps)
	for range steps {
		velocity, err := forward.Execute(ctx, pixels)
		if err != nil {
			return nil, err
		}
		for index := range pixels {
			pixels[index] += velocity[index] * delta
		}
	}
	return pixels, nil
}

func TestResidentSamplingDeviceAcceptance(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	model, err := Load(artifactDir(t))
	if err != nil {
		t.Fatalf("required sampling artifact: %v", err)
	}
	golden := loadGolden[struct {
		B, H, W int
		X, Out  []float32
	}](t, "real_forward")
	forward, err := CompileResidentForward(t.Context(), model, 0, golden.H, golden.W)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := forward.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	actual, err := forward.Execute(t.Context(), golden.X)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "independent vendor forward", actual, golden.Out, tolReal)
	const steps, seed = 3, int64(7)
	want, err := samplingHostBoundary(t.Context(), forward, steps, seed)
	if err != nil {
		t.Fatal(err)
	}
	before, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	got, err := forward.Sample(t.Context(), steps, seed)
	if err != nil {
		t.Fatal(err)
	}
	after, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "complete resident sample", got, want, tolTight)
	download := after.Execution.DeviceToHostBytes - before.Execution.DeviceToHostBytes
	t.Logf("sampling steps=%d elements=%d h2d_bytes=%d d2h_bytes=%d owned_peak_bytes=%d", steps, len(got), after.Execution.HostToDeviceBytes-before.Execution.HostToDeviceBytes, download, after.Device.PeakBytes)
	if wantBytes := uint64(len(got) * 4); download != wantBytes {
		t.Fatalf("sampling downloaded %d bytes, want one final image (%d)", download, wantBytes)
	}
	for _, count := range []int{1, 8, 3} {
		want, err := samplingHostBoundary(t.Context(), forward, count, seed)
		if err != nil {
			t.Fatal(err)
		}
		got, err := forward.Sample(t.Context(), count, seed)
		if err != nil {
			t.Fatal(err)
		}
		requireWithin(t, "changed step count", got, want, tolTight)
	}
	stable, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := forward.Sample(canceled, steps, seed); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled sample: %v", err)
	}
	if _, err := forward.Sample(t.Context(), 0, seed); err == nil {
		t.Fatal("invalid step count accepted")
	}
	control := &samplingCancellation{Context: t.Context()}
	if _, err := forward.Sample(control, 1, seed); err != nil {
		t.Fatal(err)
	}
	interrupted, interrupt := context.WithCancelCause(t.Context())
	defer interrupt(nil)
	boundary := &samplingCancellation{Context: interrupted, limit: control.checks.Load(), cancel: interrupt}
	if _, err := forward.Sample(boundary, 8, seed); !errors.Is(err, context.Canceled) {
		t.Fatalf("in-progress cancellation after %d context checks: %v", boundary.limit, err)
	}
	got, err = forward.Sample(t.Context(), steps, seed)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "post-cancellation sample", got, want, tolTight)
	recovered, err := forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if recovered.Device.CurrentBytes != stable.Device.CurrentBytes || recovered.Device.PeakBytes != stable.Device.PeakBytes {
		t.Fatalf("repeated sampling grew storage: stable=%+v recovered=%+v", stable.Device, recovered.Device)
	}
	if err := forward.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if _, err := forward.Sample(t.Context(), steps, seed); err == nil {
		t.Fatal("closed sampler accepted")
	}
}

type samplingMeasurement struct {
	Pixels                  []float32
	Durations               []time.Duration
	Compile, Initialization time.Duration
	Before, After           graphruntime.ResidentStats
}

func TestResidentSamplingSequenceAcceptance(t *testing.T) {
	cudatest.Require(t)
	directory := artifactDir(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	const resultEnvironment = "OVERGO_SAMPLING_MEASUREMENT_RESULT"
	const modeEnvironment = "OVERGO_SAMPLING_MEASUREMENT_MODE"
	if output := os.Getenv(resultEnvironment); output != "" {
		measurement := measureSamplingSequence(t, directory, os.Getenv(modeEnvironment))
		data, err := json.Marshal(measurement)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(output, data, 0600); err != nil {
			t.Fatal(err)
		}
		return
	}
	measure := func(mode string) samplingMeasurement {
		t.Helper()
		path := filepath.Join(t.TempDir(), mode+".json")
		command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v")
		command.Env = append(os.Environ(), resultEnvironment+"="+path, modeEnvironment+"="+mode)
		result, err := processmeasure.Measure(command)
		if err != nil {
			t.Fatalf("%s measured process: %v\n%s", mode, err, result.Output)
		}
		if err := testevidence.VerifyOutput("go test", string(result.Output)); err != nil {
			t.Fatal(err)
		}
		var measurement samplingMeasurement
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &measurement); err != nil {
			t.Fatal(err)
		}
		if result.PeakWorkingSetByte == 0 || len(measurement.Durations) == 0 || len(measurement.Pixels) == 0 {
			t.Fatal("required measurement is unavailable")
		}
		before, after := measurement.Before, measurement.After
		t.Logf("%s process_wall=%s process_peak_bytes=%d compile=%s initialization=%s fresh_warm_runs=%v h2d_bytes=%d d2h_bytes=%d graph_launches=%d owned_peak_bytes=%d",
			mode, result.Wall, result.PeakWorkingSetByte, measurement.Compile, measurement.Initialization, measurement.Durations,
			after.Execution.HostToDeviceBytes-before.Execution.HostToDeviceBytes,
			after.Execution.DeviceToHostBytes-before.Execution.DeviceToHostBytes,
			after.Execution.GraphLaunches-before.Execution.GraphLaunches, after.Device.PeakBytes)
		return measurement
	}
	baseline, candidate := measure("host-boundary"), measure("resident")
	requireWithin(t, "matched complete sampling", candidate.Pixels, baseline.Pixels, tolTight)
	before, after := candidate.Before, candidate.After
	repeats := len(candidate.Durations)
	if bytes := after.Execution.DeviceToHostBytes - before.Execution.DeviceToHostBytes; bytes != uint64(repeats*len(candidate.Pixels)*4) {
		t.Fatalf("resident downloads=%d, expected one image per repetition", bytes)
	}
	if after.Execution.HostToDeviceBytes-before.Execution.HostToDeviceBytes >= baseline.After.Execution.HostToDeviceBytes-baseline.Before.Execution.HostToDeviceBytes {
		t.Fatal("resident sampling did not reduce host uploads")
	}
	if after.Device.CurrentBytes != before.Device.CurrentBytes {
		t.Fatalf("warm sampling allocations grew: %d -> %d", before.Device.CurrentBytes, after.Device.CurrentBytes)
	}
	slices.Sort(baseline.Durations)
	slices.Sort(candidate.Durations)
	// A disjoint three-run envelope supports a directional timing claim. An
	// overlap is retained as no established speedup, not hidden by a best run.
	t.Logf("matched sequence: baseline_median=%s candidate_median=%s separated_speedup=%t; owned peaks are not device-wide peaks; no memory-reduction claim",
		baseline.Durations[len(baseline.Durations)/2], candidate.Durations[repeats/2], candidate.Durations[repeats-1] < baseline.Durations[0])
}

func measureSamplingSequence(t *testing.T, directory, mode string) samplingMeasurement {
	t.Helper()
	model, err := Load(directory)
	if err != nil {
		t.Fatalf("required sampling artifact: %v", err)
	}
	// Same seed/step/shape cohort and three fresh repetitions as the retained
	// generation guard. Timing compares prior CUDA sampling, not CPU sampling.
	const steps, seed, size, repeats = 2, int64(7), 64, 3
	counter := func() time.Duration {
		t.Helper()
		value, err := processmeasure.Counter()
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	started := counter()
	forward, err := CompileResidentForward(t.Context(), model, 0, size, size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := forward.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	result := samplingMeasurement{Compile: counter() - started}
	sample := func() []float32 {
		t.Helper()
		var pixels []float32
		var err error
		switch mode {
		case "host-boundary":
			pixels, err = samplingHostBoundary(t.Context(), forward, steps, seed)
		case "resident":
			pixels, err = forward.Sample(t.Context(), steps, seed)
		default:
			t.Fatalf("unknown sampling measurement mode %q", mode)
		}
		if err != nil {
			t.Fatal(err)
		}
		return pixels
	}
	started = counter()
	result.Pixels = sample()
	result.Initialization = counter() - started
	result.Before, err = forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	for range repeats {
		started := counter()
		pixels := sample()
		result.Durations = append(result.Durations, counter()-started)
		for _, pixel := range pixels {
			if math.IsNaN(float64(pixel)) || math.IsInf(float64(pixel), 0) {
				t.Fatal("non-finite sampling output")
			}
		}
		requireWithin(t, "fresh sample repeat", pixels, result.Pixels, tolTight)
	}
	result.After, err = forward.Stats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return result
}
