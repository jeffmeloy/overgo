//go:build windows

package adaptiveparity_test

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
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
	CandidateBounds struct {
		ColdStepNanos uint64 `json:"cold_step_nanos"`
		WarmStepNanos uint64 `json:"warm_step_nanos"`
	} `json:"candidate_bounds"`
}

func TestScratchTrainingLeadership(t *testing.T) {
	cudatest.Require(t)
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
	t.Logf("scratch leadership compile=%s init=%s cold=%s warm=%s/%s steps=%s peak=%d host=%d device=%d heap_base=%d heap_final=%d alloc=%d reference_steps=%s reference_peak=%d loss=%.3e weight=%.3e gradient=%.3e momentum=%.3e", compileWall, initWall, walls[0], walls[1], walls[2], stepWall, candidatePeak, hostPeak, memory.PeakBytes, baseline.HeapAlloc, finalHeap.HeapAlloc, finalHeap.TotalAlloc-baseline.TotalAlloc, time.Duration(referenceWall), referencePeak, worst, weightDelta, gradientDelta, momentumDelta)
	if worst > oracle.LossTolerance {
		t.Fatalf("scratch trajectory delta %.3e", worst)
	}
	if weightDelta > f32Bound(host.Weights) || gradientDelta > f32Bound(host.Gradients) || momentumDelta > f64AsF32Bound(host.Momentum) {
		t.Fatal("scratch final state differs from host oracle")
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
