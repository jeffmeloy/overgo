//go:build windows

package adaptiveparity_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/hybridtrain"
	"overgo/internal/optimizer"
	"overgo/internal/testutil"
)

const (
	qwen35CheckpointSHA = "c4e8dc8f885d590f146d96919562960a5d040ccc366623e986df8656539419ec"
	qwen35GradientSHA   = "76d9880f804b116fd95e3faf739bed6ad9898430e03d25bae49d5969ec1491a3"
	qwen35ServingSHA    = "6dcce666b1ebf776b9ee9e9df9107b068204affc602e94949e5c4c039894ea1a"
	qwen35AdaptiveSHA   = "214950b3b0316bcdcab38a3b95127a927c0ab5be"
	qwen35TrainingSteps = 2
)

type qwen35TrainingFixture struct {
	Cases []struct {
		Name      string   `json:"name"`
		PromptIDs []uint32 `json:"prompt_ids"`
	} `json:"cases"`
}

func TestQwen35RealTraining(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen3.5-4B-f16.gguf")
	if got := qwenFileSHA256(t, checkpoint); got != qwen35CheckpointSHA {
		t.Fatalf("Qwen3.5 checkpoint identity=%s, want %s", got, qwen35CheckpointSHA)
	}
	if got := qwenFileSHA256(t, filepath.Join(root, "fixtures", "qwen35_hybrid_grad_golden.json")); got != qwen35GradientSHA {
		t.Fatalf("Qwen3.5 gradient oracle identity=%s, want %s", got, qwen35GradientSHA)
	}
	servingPath := filepath.Join(root, "fixtures", "qwen35_4b_serving_golden.json")
	if got := qwenFileSHA256(t, servingPath); got != qwen35ServingSHA {
		t.Fatalf("Qwen3.5 serving data identity=%s, want %s", got, qwen35ServingSHA)
	}
	raw, err := os.ReadFile(servingPath)
	if err != nil {
		t.Fatal(err)
	}
	var serving qwen35TrainingFixture
	if err := json.Unmarshal(raw, &serving); err != nil || len(serving.Cases) == 0 || serving.Cases[0].Name != "capital" || len(serving.Cases[0].PromptIDs) < 2 {
		t.Fatalf("Qwen3.5 serving data is invalid: %v", err)
	}

	loaded := time.Now()
	trained, err := hybridtrain.LoadRecurrentLayerArtifact(
		context.Background(), checkpoint, 0, serving.Cases[0].PromptIDs,
	)
	if err != nil {
		t.Fatal(err)
	}
	loadWall := time.Since(loaded)
	if !trained.Program().ID().Valid() || trained.MatrixParamCount() == 0 || trained.VectorParamCount() == 0 {
		t.Fatal("Qwen3.5 training program is incomplete")
	}
	runtime.GC()

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	if err := worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	trajectory, residency, err := trained.TrainDeviceResident(worker, qwen35TrainingSteps, optimizer.Config{
		BaseLearningRate: 1e-4,
		Momentum:         0.9,
		Schedule:         optimizer.ScheduleConstant,
	})
	trainWall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	testutil.RequireFiniteDecrease(t, "Qwen3.5 real-layer trajectory", trajectory, qwen35TrainingSteps)
	if residency.WeightUploads != 1 || residency.MomentumUploads != 1 || residency.MomentumReads != 0 ||
		residency.WeightReads != 0 || residency.FinalWeightRead != 1 || residency.GradUploads != 2 {
		t.Fatalf("Qwen3.5 residency differs: %+v", residency)
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Qwen3.5-4B recurrent layer 0: params=%d+%d loss=%v load=%s train=%s peak=%.3fGiB program=%s",
		trained.MatrixParamCount(), trained.VectorParamCount(), trajectory, loadWall, trainWall,
		float64(memory.PeakBytes)/(1<<30), trained.Program().ID())
	t.Logf("adaptive_new %s has no Qwen3.5 training oracle; result is Overgo-only, not parity evidence", qwen35AdaptiveSHA)
}
