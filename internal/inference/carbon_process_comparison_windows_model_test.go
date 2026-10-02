//go:build windows && modeltest

package inference

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testprocess"
	"overgo/internal/testutil"
)

const carbonProcessRuns = 5
const carbonWarmParityDivisor = 10

// The reference is adaptive_new's Go module at the commit the recorded Carbon
// comparison measured, with the reference probe this package owns added: the
// commit holds no process probe of its own.
const (
	carbonReferenceCommit = "214950b3b0316bcdcab38a3b95127a927c0ab5be"
	carbonReferenceProbe  = "testdata/adaptive_carbon_process_probe_test.go"
)

var carbonProcessMarker = regexp.MustCompile(`CARBON_PROCESS_PROBE wall_ns=([0-9]+).*warm_ns=([0-9]+).*device_peak=([0-9]+)`)
var carbonCandidatePhaseMarker = regexp.MustCompile(`resolve_ns=([0-9]+) open_ns=([0-9]+) generate_ns=([0-9]+)`)
var carbonProcessLineMarker = regexp.MustCompile(`CARBON_PROCESS_PROBE[^\r\n]*`)

func TestCarbonColdLifecyclePhases(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(t.TempDir(), "repodb")
	store, err := overgodb.Open(storePath)
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
	probe, err := os.ReadFile(carbonReferenceProbe)
	if err != nil {
		t.Fatalf("UNAVAILABLE: adaptive Carbon reference probe absent: adaptive commit %s holds no TestCarbonServingProcessProbe: %v",
			carbonReferenceCommit, err)
	}
	adaptiveGo := testprocess.PinnedTree(t, filepath.Dir(roots.Models), carbonReferenceCommit, "go")
	if err := os.WriteFile(filepath.Join(adaptiveGo, "cmd", "adaptive-gpt-server", filepath.Base(carbonReferenceProbe)), probe, 0o644); err != nil {
		t.Fatal(err)
	}
	// Adaptive serves Carbon from the checkpoint the GGUF is converted from,
	// through the active recipe its catalog holds: its own seeder writes that
	// catalog for the models directory into a fresh store, and the reference
	// child reads it through adaptive's store redirect, as adaptive's commit
	// gate serves a candidate catalog.
	adaptiveModel := filepath.Join(roots.Models, "Carbon-500M")
	adaptiveStore := filepath.Join(t.TempDir(), "repodb")
	seed := exec.Command("go", "run", "./cmd/repodb", "seed-model-profiles",
		"-store", adaptiveStore, "-root", filepath.Dir(adaptiveModel), "-base", ".")
	seed.Dir = adaptiveGo
	if output, err := seed.CombinedOutput(); err != nil {
		t.Fatalf("seed adaptive Carbon catalog: %v: %s", err, output)
	}
	t.Setenv("ADAPTIVE_MODEL_STORE_REDIRECT_FROM", filepath.Join(filepath.Dir(adaptiveModel), "docs", "repodb"))
	t.Setenv("ADAPTIVE_MODEL_STORE_REDIRECT_TO", adaptiveStore)
	t.Setenv("OVERGO_CARBON_ADAPTIVE_MODEL", adaptiveModel)
	t.Setenv("OVERGO_CARBON_GOLDEN", testutil.FixturePath(t, "carbon_serving_golden.json"))
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-carbon.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-carbon.test.exe")
	testprocess.BuildTestBinary(t, root, candidateBinary, "modeltest", "./internal/inference")
	testprocess.BuildTestBinary(t, adaptiveGo, referenceBinary, "cuda", "./cmd/adaptive-gpt-server")

	candidate := make([]processmeasure.Result, 0, carbonProcessRuns)
	reference := make([]processmeasure.Result, 0, carbonProcessRuns)
	measureCandidate := func() {
		candidate = append(candidate, testprocess.MeasureTestProcesses(t, 1, root, candidateBinary,
			"TestCarbonServingProcessProbe", "CARBON_PROCESS_PROBE")[0])
	}
	measureReference := func() {
		reference = append(reference, testprocess.MeasureTestProcesses(t, 1,
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
	resolveWall, openWall, generationWall := carbonCandidatePhaseMedians(t, candidate)
	candidateHost, referenceHost := testprocess.MedianProcessPeak(candidate), testprocess.MedianProcessPeak(reference)
	logCarbonProbeSamples(t, "candidate", candidate)
	logCarbonProbeSamples(t, "reference", reference)
	t.Logf("Carbon candidate=%s reference=%s", testprocess.FormatProcessMeasurements(candidate), testprocess.FormatProcessMeasurements(reference))
	t.Logf("Carbon lifecycle candidate=%s/%0.3fMiB reference=%s/%0.3fMiB",
		candidateWall, testprocess.MiB(candidateDevice), referenceWall, testprocess.MiB(referenceDevice))
	t.Logf("Carbon warm generation candidate=%s reference=%s", candidateWarm, referenceWarm)
	t.Logf("Carbon candidate cold phases resolve=%s open=%s generation=%s", resolveWall, openWall, generationWall)
	if candidateDevice >= referenceDevice {
		t.Fatalf("Carbon device peak %.3f MiB does not beat adaptive %.3f MiB",
			testprocess.MiB(candidateDevice), testprocess.MiB(referenceDevice))
	}
	if candidateHost >= referenceHost {
		t.Fatalf("Carbon process peak %.3f MiB does not beat adaptive %.3f MiB",
			testprocess.MiB(candidateHost), testprocess.MiB(referenceHost))
	}
	if 2*candidateWall >= 3*referenceWall {
		t.Fatalf("Carbon lifecycle wall %s exceeds 1.5x adaptive %s", candidateWall, referenceWall)
	}
	if candidateWarm > referenceWarm+referenceWarm/carbonWarmParityDivisor {
		t.Fatalf("Carbon warm generation %s exceeds 10%% parity band around adaptive %s", candidateWarm, referenceWarm)
	}
}

func logCarbonProbeSamples(t testing.TB, label string, results []processmeasure.Result) {
	t.Helper()
	for index, result := range results {
		marker := carbonProcessLineMarker.FindString(string(result.Output))
		if marker == "" {
			marker = "marker absent"
		}
		t.Logf("Carbon %s process %d: %s", label, index+1, marker)
	}
}

func carbonCandidatePhaseMedians(t testing.TB, results []processmeasure.Result) (time.Duration, time.Duration, time.Duration) {
	t.Helper()
	resolveWalls := make([]time.Duration, len(results))
	openWalls := make([]time.Duration, len(results))
	generationWalls := make([]time.Duration, len(results))
	for index, result := range results {
		match := carbonCandidatePhaseMarker.FindSubmatch(result.Output)
		if len(match) != 4 {
			t.Fatalf("Carbon candidate phase marker is malformed: %s", result.Output)
		}
		for field, target := range []*time.Duration{&resolveWalls[index], &openWalls[index], &generationWalls[index]} {
			value, err := strconv.ParseUint(string(match[field+1]), 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			*target = time.Duration(value)
		}
	}
	slices.Sort(resolveWalls)
	slices.Sort(openWalls)
	slices.Sort(generationWalls)
	middle := len(results) / 2
	return resolveWalls[middle], openWalls[middle], generationWalls[middle]
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
