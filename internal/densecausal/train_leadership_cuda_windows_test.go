//go:build windows

package densecausal

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

const (
	// Adaptive retained: 2.018 s/step (9c80f6ab0); 15.06 GB peak (4faffd8f2).
	carbonStepWallRatchet = 1800 * time.Millisecond
	carbonPeakRatchet     = uint64(6 << 30)
)

var carbonAdaptiveCausalWindows = [][]int{
	{151669, 154337, 154656, 154674, 152581, 153742, 154236, 154503, 153824, 152624, 153810, 153761, 155384, 154435, 155623, 153169, 154707, 152390, 154190, 153393, 155460, 153561, 152593, 153068, 154772, 153675, 154235, 151812, 154071, 152533, 155065},
	{151669, 154192, 153702, 151805, 154333, 154612, 151945, 154411, 153097, 152426, 152484, 152557, 153173, 154518, 155591, 152212, 152421, 153636, 155578, 153856, 154694, 151896, 152427, 151796, 155370, 153522, 152602, 155575, 152807, 154535, 154708},
	{151669, 154938, 151700, 151940, 153476, 153803, 151738, 153283, 152040, 152248, 154493, 151684, 151807, 151880, 153848, 153661, 155558, 152844, 153565, 154061, 154045, 153475, 154069, 152452, 154596, 151860, 155749, 154631, 154340, 153470, 154077},
	{151669, 154113, 154090, 153886, 152057, 155395, 153055, 153759, 152006, 151950, 154247, 152619, 154794, 152482, 152335, 154318, 155670, 153102, 155314, 153361, 153649, 153350, 153330, 153358, 153406, 152110, 151686, 155020, 153894, 154070, 151826},
}

func TestCarbonMatchedAdaptiveLeadership(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_DENSE_TRAIN_BASELINE") != "1" {
		t.Skip("set OVERGO_DENSE_TRAIN_BASELINE=1 for matched Carbon training")
	}
	model, err := Load(artifactDir(t, "Carbon-500M"))
	if err != nil {
		t.Fatal(err)
	}
	warmModel, err := Load(artifactDir(t, "Carbon-500M"))
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	if _, err := warmModel.TrainDeviceResidentFrozenLexicalBatches(worker, carbonAdaptiveCausalWindows[:1], 1.33179e-5, 0.95); err != nil {
		t.Fatalf("Carbon warm-up: %v", err)
	}
	if err := worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	trajectory, loopWall, err := model.MeasureDeviceResidentFrozenLexicalBatches(worker, carbonAdaptiveCausalWindows, 1.33179e-5, 0.95)
	totalWall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	const (
		adaptiveInitial = 8.43767
		adaptiveWall    = 6816 * time.Millisecond
		adaptivePeak    = uint64(12315818721)
	)
	if delta := math.Abs(trajectory[0] - adaptiveInitial); delta > 0.02 {
		t.Fatalf("initial loss %.6f differs from adaptive %.6f by %.3e", trajectory[0], adaptiveInitial, delta)
	}
	if trajectory[len(trajectory)-1] > trajectory[0]*1.05 {
		t.Fatalf("matched Carbon loss %.6f -> %.6f", trajectory[0], trajectory[len(trajectory)-1])
	}
	if loopWall >= adaptiveWall {
		t.Fatalf("matched Carbon loop wall %s does not beat adaptive %s", loopWall, adaptiveWall)
	}
	if memory.PeakBytes >= adaptivePeak {
		t.Fatalf("matched Carbon peak %.3fGiB does not beat adaptive %.3fGiB", float64(memory.PeakBytes)/(1<<30), float64(adaptivePeak)/(1<<30))
	}
	t.Logf("matched Carbon causal: windows=%d seq=%d loss %.6f->%.6f loop=%.3fs total=%.3fs peak=%.3fGiB adaptive_loop=%.3fs/%.3fGiB",
		len(trajectory), len(carbonAdaptiveCausalWindows[0]), trajectory[0], trajectory[len(trajectory)-1], loopWall.Seconds(), totalWall.Seconds(), float64(memory.PeakBytes)/(1<<30), adaptiveWall.Seconds(), float64(adaptivePeak)/(1<<30))
}

func TestCarbonResidentTrainingLeadership(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_DENSE_TRAIN_BASELINE") != "1" {
		t.Skip("set OVERGO_DENSE_TRAIN_BASELINE=1 for real Carbon training")
	}
	model, err := Load(artifactDir(t, "Carbon-500M"))
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	tokens := make([]int, 64)
	tokens[0] = 1
	for index := 1; index < len(tokens); index++ {
		tokens[index] = 151669 + index%64
	}
	embed := model.Weights["model.embed_tokens.weight"]
	probes := []int{0, len(embed) / 3, len(embed) - 1}
	before := make([]float32, len(probes))
	for index, probe := range probes {
		before[index] = embed[probe]
	}
	if err := worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	trajectory, err := model.TrainDeviceResidentFrozenLexical(worker, tokens, 4, 0, 0.95)
	wall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for index, loss := range trajectory {
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("loss[%d]=%g", index, loss)
		}
	}
	if trajectory[len(trajectory)-1] > trajectory[0]*1.05 {
		t.Fatalf("Carbon loss %.6f -> %.6f exceeds bounded trajectory", trajectory[0], trajectory[len(trajectory)-1])
	}
	stepWall := wall / time.Duration(len(trajectory))
	if stepWall >= carbonStepWallRatchet {
		t.Fatalf("Carbon step wall %s does not beat %s ratchet", stepWall, carbonStepWallRatchet)
	}
	if memory.PeakBytes >= carbonPeakRatchet {
		t.Fatalf("Carbon peak %.3fGiB does not beat %.3fGiB ratchet", float64(memory.PeakBytes)/(1<<30), float64(carbonPeakRatchet)/(1<<30))
	}
	for index, probe := range probes {
		if embed[probe] != before[index] {
			t.Fatalf("frozen lexical probe %d changed", probe)
		}
	}
	t.Logf("Carbon resident Muon: seq=%d steps=%d loss %.6f->%.6f wall=%.3fs (%.3fs/step) peak=%.3fGiB",
		len(tokens), len(trajectory), trajectory[0], trajectory[len(trajectory)-1], wall.Seconds(), stepWall.Seconds(), float64(memory.PeakBytes)/(1<<30))
}
