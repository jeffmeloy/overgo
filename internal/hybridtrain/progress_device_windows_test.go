package hybridtrain

import (
	"fmt"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/optimizer"
)

func deviceSegmentTrainers(worker *device.Worker) []segmentTrainer {
	return []segmentTrainer{
		func(m *Model, n int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
			return m.TrainHostMasterStreamed(worker, n, cfg, options)
		},
		func(m *Model, n int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
			return m.TrainHostMasterLayerStreamed(worker, n, cfg, options)
		},
		func(m *Model, n int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
			losses, counts, err := m.TrainDeviceResident(worker, n, cfg, options)
			reads := 0
			if options.Checkpoint != nil {
				reads = 1
			}
			if err == nil && (counts.Steps != len(losses) || counts.MomentumReads != reads) {
				return nil, fmt.Errorf("resident progress counters: %+v", counts)
			}
			return losses, err
		},
	}
}

func TestHybridOptimizerResumeDevice(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	for index, train := range deviceSegmentTrainers(worker) {
		t.Logf("backend %d", index)
		verifyOptimizerResume(t, train)
		verifyOptimizerStop(t, train)
	}
	stats, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if stats.CurrentBytes != 0 {
		t.Fatalf("retained %d device bytes", stats.CurrentBytes)
	}
}
