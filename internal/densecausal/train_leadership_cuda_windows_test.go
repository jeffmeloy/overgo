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
