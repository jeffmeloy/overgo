//go:build windows && modeltest

package inference

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

const carbonProcessRuns = 5

var carbonProcessMarker = regexp.MustCompile(`CARBON_PROCESS_PROBE wall_ns=([0-9]+).*warm_ns=([0-9]+).*device_peak=([0-9]+)`)

func TestCarbonWarmGenerationParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(t.TempDir(), "repodb")
	store, err := repodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Carbon-500M-f16-ropefix.gguf")
	if err := servingtest.PublishActiveGGUFWithPolicy(
		context.Background(), store, modelPath, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OVERGO_CARBON_REPODB", storePath)
	adaptiveGo := filepath.Join(root, "build", "adaptive-carbon-214950b", "go")
	if _, err := os.Stat(filepath.Join(adaptiveGo, "go.mod")); err != nil {
		t.Fatalf("UNAVAILABLE: retained adaptive Carbon source absent at %s", adaptiveGo)
	}
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-carbon.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-carbon.test.exe")
	testutil.BuildTestBinary(t, root, candidateBinary, "modeltest", "./internal/inference")
	testutil.BuildTestBinary(t, adaptiveGo, referenceBinary, "cuda", "./cmd/adaptive-gpt-server")

	candidate := make([]processmeasure.Result, 0, carbonProcessRuns)
	reference := make([]processmeasure.Result, 0, carbonProcessRuns)
	measureCandidate := func() {
		candidate = append(candidate, testutil.MeasureTestProcesses(t, 1, root, candidateBinary,
			"TestCarbonServingProcessProbe", "CARBON_PROCESS_PROBE")[0])
	}
	measureReference := func() {
		reference = append(reference, testutil.MeasureTestProcesses(t, 1,
			filepath.Join(adaptiveGo, "cmd", "adaptive-gpt-server"), referenceBinary,
			"TestCarbonServingProcessProbe", "CARBON_PROCESS_PROBE")[0])
	}
	for index := range carbonProcessRuns {
		if index%2 == 0 {
			measureCandidate()
			measureReference()
		} else {
			measureReference()
			measureCandidate()
		}
	}
	candidateWall, candidateWarm, candidateDevice := carbonProbeMedians(t, candidate)
	referenceWall, referenceWarm, referenceDevice := carbonProbeMedians(t, reference)
	candidateHost, referenceHost := testutil.MedianProcessPeak(candidate), testutil.MedianProcessPeak(reference)
	t.Logf("Carbon candidate=%s reference=%s", testutil.FormatProcessMeasurements(candidate), testutil.FormatProcessMeasurements(reference))
	t.Logf("Carbon lifecycle candidate=%s/%0.3fMiB reference=%s/%0.3fMiB",
		candidateWall, testutil.MiB(candidateDevice), referenceWall, testutil.MiB(referenceDevice))
	t.Logf("Carbon warm generation candidate=%s reference=%s", candidateWarm, referenceWarm)
	if candidateDevice >= referenceDevice {
		t.Fatalf("Carbon device peak %.3f MiB does not beat adaptive %.3f MiB",
			testutil.MiB(candidateDevice), testutil.MiB(referenceDevice))
	}
	if candidateHost >= referenceHost {
		t.Fatalf("Carbon process peak %.3f MiB does not beat adaptive %.3f MiB",
			testutil.MiB(candidateHost), testutil.MiB(referenceHost))
	}
	if candidateWall >= 2*referenceWall {
		t.Fatalf("Carbon lifecycle wall %s exceeds 2x adaptive %s", candidateWall, referenceWall)
	}
	if candidateWarm > referenceWarm+referenceWarm/50 {
		t.Fatalf("Carbon warm generation %s exceeds 2%% parity band around adaptive %s", candidateWarm, referenceWarm)
	}
}

func carbonProbeMedians(t testing.TB, results []processmeasure.Result) (time.Duration, time.Duration, uint64) {
	t.Helper()
	walls := make([]time.Duration, len(results))
	warms := make([]time.Duration, len(results))
	peaks := make([]uint64, len(results))
	for index, result := range results {
		match := carbonProcessMarker.FindSubmatch(result.Output)
		if len(match) != 4 {
			t.Fatalf("Carbon process marker is malformed: %s", result.Output)
		}
		wall, err := strconv.ParseUint(string(match[1]), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		warm, err := strconv.ParseUint(string(match[2]), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		peak, err := strconv.ParseUint(string(match[3]), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		walls[index], warms[index], peaks[index] = time.Duration(wall), time.Duration(warm), peak
	}
	slices.Sort(walls)
	slices.Sort(warms)
	slices.Sort(peaks)
	return walls[len(walls)/2], warms[len(warms)/2], peaks[len(peaks)/2]
}
