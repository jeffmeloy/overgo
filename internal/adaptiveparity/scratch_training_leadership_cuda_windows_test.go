//go:build windows

package adaptiveparity_test

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/adaptiveparity"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/scratchmodel"
	"overgo/internal/strictjson"
)

type scratchPerformanceEvidence struct {
	Schema           string `json:"schema"`
	SourceCommit     string `json:"source_commit"`
	SourceEntrypoint string `json:"source_entrypoint"`
	SourceOptions    string `json:"source_options"`
	OracleSHA256     string `json:"oracle_sha256"`
	Environment      struct {
		Go                string `json:"go"`
		OSArch            string `json:"os_arch"`
		CPU               string `json:"cpu"`
		LogicalProcessors int    `json:"logical_processors"`
	} `json:"environment"`
	Runs []struct {
		WallNanos       uint64 `json:"wall_nanos"`
		PeakHeapBytes   uint64 `json:"peak_heap_bytes"`
		TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	} `json:"runs"`
	CandidateObservations struct {
		ProcessInitialization []uint64 `json:"process_initialization_nanos"`
		DriverPreparation     []uint64 `json:"driver_preparation_nanos"`
		ModelInitialization   []uint64 `json:"model_initialization_nanos"`
		ProgramPreparation    []uint64 `json:"program_preparation_nanos"`
		ColdStep              []uint64 `json:"cold_step_nanos"`
		WarmStep              []uint64 `json:"warm_step_nanos"`
	} `json:"candidate_observations"`
	CandidateBounds struct {
		ProcessInitializationNanos uint64 `json:"process_initialization_nanos"`
		DriverPreparationNanos     uint64 `json:"driver_preparation_nanos"`
		ModelInitializationNanos   uint64 `json:"model_initialization_nanos"`
		ProgramPreparationNanos    uint64 `json:"program_preparation_nanos"`
		ColdStepNanos              uint64 `json:"cold_step_nanos"`
		WarmStepNanos              uint64 `json:"warm_step_nanos"`
	} `json:"candidate_bounds"`
}

func TestScratchTrainingLeadership(t *testing.T) {
	if os.Getenv(scratchProcessReady) == "1" {
		fmt.Println(scratchProcessReady)
		return
	}
	cudatest.Require(t)
	processInitialization := measureScratchProcessInitialization(t)
	oracleBytes, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	oracle, _, err := adaptiveparity.NormalizeScratchOracle(oracleBytes)
	if err != nil {
		t.Fatal(err)
	}
	evidenceBytes, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_performance.json"))
	if err != nil {
		t.Fatal(err)
	}
	var evidence scratchPerformanceEvidence
	if err := strictjson.DecodeBytes(evidenceBytes, &evidence); err != nil {
		t.Fatal(err)
	}
	oracleDigest := sha256.Sum256(oracleBytes)
	if evidence.Schema != "overgo/adaptive-scratch-performance/v1" || evidence.SourceCommit != oracle.SourceCommit ||
		evidence.SourceEntrypoint != "go/training/run_from_docs.go:RunFromDocsRich" || evidence.SourceOptions != "seed=7;steps=3;batch_size_override=1" || evidence.Environment.CPU == "" ||
		evidence.OracleSHA256 != hex.EncodeToString(oracleDigest[:]) || evidence.Environment.Go != runtime.Version() ||
		evidence.Environment.OSArch != runtime.GOOS+"/"+runtime.GOARCH || evidence.Environment.LogicalProcessors != runtime.NumCPU() || len(evidence.Runs) < 5 {
		t.Fatal("adaptive scratch performance evidence fingerprint differs")
	}
	observations := [][]uint64{
		evidence.CandidateObservations.ProcessInitialization,
		evidence.CandidateObservations.DriverPreparation,
		evidence.CandidateObservations.ModelInitialization,
		evidence.CandidateObservations.ProgramPreparation,
		evidence.CandidateObservations.ColdStep,
		evidence.CandidateObservations.WarmStep,
	}
	bounds := []uint64{
		evidence.CandidateBounds.ProcessInitializationNanos,
		evidence.CandidateBounds.DriverPreparationNanos,
		evidence.CandidateBounds.ModelInitializationNanos,
		evidence.CandidateBounds.ProgramPreparationNanos,
		evidence.CandidateBounds.ColdStepNanos,
		evidence.CandidateBounds.WarmStepNanos,
	}
	for index := range observations {
		bound, ok := lifecycleEnvelopeBound(observations[index])
		if len(observations[index]) < 5 || !ok || bounds[index] != bound {
			t.Fatal("scratch lifecycle bound differs from evidence envelope")
		}
	}
	referenceWall, referencePeak := uint64(math.MaxUint64), uint64(math.MaxUint64)
	for _, run := range evidence.Runs {
		if run.WallNanos == 0 || run.PeakHeapBytes == 0 || run.TotalAllocBytes < run.PeakHeapBytes {
			t.Fatal("adaptive scratch performance evidence run invalid")
		}
		referenceWall = min(referenceWall, run.WallNanos)
		referencePeak = min(referencePeak, run.PeakHeapBytes)
	}

	compileStarted := time.Now()
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps})
	if err != nil {
		t.Fatal(err)
	}
	compileWall := time.Since(compileStarted)
	host, err := construction.TrainShared(oracle.Steps)
	if err != nil {
		t.Fatal(err)
	}
	initStarted := time.Now()
	trainer, err := scratchmodel.NewResidentTrainer(construction, oracle.Steps)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	initWall := time.Since(initStarted)
	lifecycle, err := trainer.Lifecycle()
	if err != nil {
		t.Fatal(err)
	}
	if err := trainer.ResetPeakMemory(); err != nil {
		t.Fatal(err)
	}

	runtime.GC()
	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)
	var peak atomic.Uint64
	peak.Store(baseline.HeapAlloc)
	done := make(chan struct{})
	go sampleHeap(done, &peak)
	losses := make([]float64, oracle.Steps)
	walls := make([]time.Duration, oracle.Steps)
	for step := range oracle.Steps {
		tokens, err := construction.Tokens(oracle.Train[step%len(oracle.Train)])
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		losses[step], err = trainer.Step(tokens, step+1)
		walls[step] = time.Since(started)
		if err != nil {
			t.Fatal(err)
		}
	}
	close(done)
	var finalHeap runtime.MemStats
	runtime.ReadMemStats(&finalHeap)
	validationTokens, err := construction.Tokens(oracle.Validation[0])
	if err != nil {
		t.Fatal(err)
	}
	validation, err := trainer.Evaluate(validationTokens)
	if err != nil {
		t.Fatal(err)
	}
	memory, err := trainer.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	weights, gradients, momentum, err := trainer.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	hostPeak := peak.Load() - baseline.HeapAlloc
	candidatePeak := memory.PeakBytes + hostPeak
	stepWall := walls[0] + walls[1] + walls[2]
	worst := math.Abs(validation - oracle.FinalValLoss)
	for index := range losses {
		worst = max(worst, math.Abs(losses[index]-oracle.LossHistory[index]))
	}
	weightDelta := maxF32Delta(weights, host.Weights)
	gradientDelta := maxF32Delta(gradients, host.Gradients)
	momentumDelta := maxF32F64Delta(momentum, host.Momentum)
	t.Logf("scratch leadership process=%s compile=%s init=%s driver=%s model=%s program=%s cold=%s warm=%s/%s steps=%s peak=%d host=%d device=%d heap_base=%d heap_final=%d alloc=%d reference_steps=%s reference_peak=%d loss=%.3e weight=%.3e gradient=%.3e momentum=%.3e", processInitialization, compileWall, initWall, lifecycle.DriverPreparation, lifecycle.ModelInitialization, lifecycle.ProgramPreparation, walls[0], walls[1], walls[2], stepWall, candidatePeak, hostPeak, memory.PeakBytes, baseline.HeapAlloc, finalHeap.HeapAlloc, finalHeap.TotalAlloc-baseline.TotalAlloc, time.Duration(referenceWall), referencePeak, worst, weightDelta, gradientDelta, momentumDelta)
	if worst > oracle.LossTolerance {
		t.Fatalf("scratch trajectory delta %.3e", worst)
	}
	if weightDelta > f32Bound(host.Weights) || gradientDelta > f32Bound(host.Gradients) || momentumDelta > f64AsF32Bound(host.Momentum) {
		t.Fatal("scratch final state differs from host oracle")
	}
	if uint64(processInitialization) > evidence.CandidateBounds.ProcessInitializationNanos ||
		uint64(lifecycle.DriverPreparation) > evidence.CandidateBounds.DriverPreparationNanos ||
		uint64(lifecycle.ModelInitialization) > evidence.CandidateBounds.ModelInitializationNanos ||
		uint64(lifecycle.ProgramPreparation) > evidence.CandidateBounds.ProgramPreparationNanos {
		t.Fatal("scratch preparation lifecycle exceeds evidence envelope")
	}
	if uint64(walls[0]) > evidence.CandidateBounds.ColdStepNanos || uint64(walls[1]) > evidence.CandidateBounds.WarmStepNanos || uint64(walls[2]) > evidence.CandidateBounds.WarmStepNanos {
		t.Fatalf("scratch lifecycle exceeds cold/warm ratchet: %v", walls)
	}
	if uint64(stepWall) >= referenceWall {
		t.Fatalf("scratch steps %s do not beat adaptive %s", stepWall, time.Duration(referenceWall))
	}
	if candidatePeak >= referencePeak {
		t.Fatalf("scratch peak %d does not beat adaptive %d", candidatePeak, referencePeak)
	}
}

func lifecycleEnvelopeBound(observations []uint64) (uint64, bool) {
	if len(observations) == 0 || observations[0] == 0 {
		return 0, false
	}
	low, high := observations[0], observations[0]
	for _, observation := range observations[1:] {
		if observation == 0 {
			return 0, false
		}
		low, high = min(low, observation), max(high, observation)
	}
	spread := high - low
	if high > math.MaxUint64-spread {
		return 0, false
	}
	return high + spread, true
}

const scratchProcessReady = "OVERGO_SCRATCH_PROCESS_READY"

func measureScratchProcessInitialization(t *testing.T) time.Duration {
	t.Helper()
	command := exec.Command(os.Args[0], "-test.run=^TestScratchTrainingLeadership$", "-test.count=1", "-test.v")
	command.Env = append(os.Environ(), scratchProcessReady+"=1")
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	command.Stderr = &stderr
	started := time.Now()
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	var initialization time.Duration
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		if scanner.Text() == scratchProcessReady {
			initialization = time.Since(started)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("scratch process probe: %v: %s", err, stderr.String())
	}
	if initialization <= 0 {
		t.Fatal("scratch process probe marker absent")
	}
	return initialization
}

func maxF32Delta(left, right []float32) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	var worst float64
	for index := range left {
		worst = max(worst, math.Abs(float64(left[index]-right[index])))
	}
	return worst
}

func maxF32F64Delta(left []float32, right []float64) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	var worst float64
	for index := range left {
		worst = max(worst, math.Abs(float64(left[index])-right[index]))
	}
	return worst
}

func f32Bound(reference []float32) float64 {
	scale := float64(1)
	for _, value := range reference {
		scale = max(scale, math.Abs(float64(value)))
	}
	return math.Sqrt(float64(math.Nextafter32(1, 2)-1)) * scale
}

func f64AsF32Bound(reference []float64) float64 {
	scale := float64(1)
	for _, value := range reference {
		scale = max(scale, math.Abs(value))
	}
	return math.Sqrt(float64(math.Nextafter32(1, 2)-1)) * scale
}

func sampleHeap(done <-chan struct{}, peak *atomic.Uint64) {
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			var sample runtime.MemStats
			runtime.ReadMemStats(&sample)
			for prior := peak.Load(); sample.HeapAlloc > prior && !peak.CompareAndSwap(prior, sample.HeapAlloc); prior = peak.Load() {
			}
		}
	}
}
