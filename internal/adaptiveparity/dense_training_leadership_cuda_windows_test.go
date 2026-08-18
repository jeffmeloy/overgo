//go:build windows

package adaptiveparity_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/testutil"
)

const (
	qwenCheckpointSHA  = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
	qwenConfigSHA      = "ce7908dfd631cd1f713a65cbea3591d32acaab58d438127d70c5dfc5ce1fc1f0"
	qwenCorpusSHA      = "e8991b6e58d79e2d8dc6f22175e486a90c269f88bc1eab2718946f201ab32c8a"
	qwenProtocolSHA    = "9323c549adad08ed046f366977d236f891914e6076b047c67dd2e3a762dcde30"
	qwenProtocolFormat = "source-corpus:adaptive-de547a363;model:qwen2.5-0.5b;seq:%d;optimizer:muon-4x2-tf32;lr:%g;mu:%g;frozen:lexical;steps:%d"
	qwenAdaptiveWarm   = 505 * time.Millisecond

	qwenLearningRate       = 1e-4
	qwenMomentum           = 0.9
	qwenInitialLoss        = 4.59284
	qwenInitialLossLimit   = 0.01
	qwenUpdatedLossMinimum = 2.8
	qwenUpdatedLossMaximum = 3.1
	qwenPeakMemoryGiB      = 7
	bytesPerGiB            = 1 << 30
)

const (
	qwenInitialStep = iota
	qwenUpdatedStep
	qwenTrainingSteps
)

var qwenAdaptiveProfileTokens = []int{
	4913, 1028, 3252, 17, 15, 17, 21, 12, 15, 21, 12, 16, 24, 2198, 21621, 3252, 334, 7012, 19811, 12, 2448, 12, 21, 55, 17, 12010, 2716, 4835, 12, 54326, 12010, 2336,
	13, 334, 2198, 423, 8477, 3252, 38, 33793, 16151, 3027, 80285, 79854, 26558, 389, 279, 9867, 220, 23, 10, 17, 3043, 9700, 26, 279, 17201, 220, 21, 10, 17, 4269, 24376,
	4641, 3468, 26511, 7902, 49615, 11, 323, 264, 10735, 7616, 1431, 27934, 48916, 4269, 9700, 33638, 47891, 1330, 491, 3252, 9432, 54809, 49615, 17551, 279, 7205, 4269, 3398, 25, 3393,
	36930, 3027, 80285, 9296, 1397, 58642, 1671, 91931, 38, 33793, 33951, 5485, 1614, 31633, 5047, 21004, 13, 15, 27149, 3752, 1517, 11354, 49453, 25152, 58, 19, 21, 16, 24, 60, 33638, 553,
	220, 15, 13, 15, 15, 22, 22, 20, 23, 18, 24, 1124, 84, 15, 15, 18, 68, 220, 15, 13, 15, 15, 20, 2348, 23577, 2589, 27781, 13, 4636, 97767, 4269, 1182, 311, 220, 23, 10, 17, 11, 3393,
	36930, 3027, 80285, 79854, 6608, 7124, 20800, 53302, 8796, 11, 3393, 36930, 3027, 80285, 79854, 48849, 82, 86433, 4562, 2354, 35576, 36219, 11, 323, 3393, 36930, 3027, 80285, 9296, 1397, 58642, 1671, 91931, 38,
	33793, 33951, 1494, 13, 42522, 18967, 220, 21, 10, 17, 8458, 27259, 26, 4269, 3880, 264, 3468, 14859, 2144, 4846, 349, 1573, 24376, 1189, 532, 10952, 606, 25, 38193, 62772, 198, 4684, 25, 96448, 19569, 32135,
	369, 47132, 1889, 417, 13, 9947, 21324, 374, 279, 1172, 1909, 11591, 50494, 8315, 7600, 7329, 624, 44364, 2, 96448, 13704, 367, 198, 198, 565, 23220, 198, 198, 12, 7854, 825, 2266, 71186, 20178, 28894,
	323, 825, 20178, 1614, 624, 12, 13655, 1846, 8186, 9489, 25, 279, 28894, 1969, 975, 369, 279, 15867, 594, 198, 220, 10337, 71186, 9867, 1614, 323, 369, 7248, 9250, 27950, 50777, 35414, 4119, 280, 220, 458, 9250,
	28244, 5452, 3164, 374, 17484, 2615, 11, 537, 11064, 429, 9867, 198, 220, 95254, 65961, 2022, 374, 27956, 624, 12, 2657, 63306, 512, 3004, 525, 1172, 1565, 21378, 7808, 1565, 16475, 29725, 7808, 323, 1565, 17269,
	18639, 12, 20094, 770, 374, 14257, 11, 11062, 11, 476, 20975, 33056, 438, 458, 1787, 10963, 304, 1565, 44, 1890, 19100, 4323, 18639, 12, 16554, 11853, 369, 1614, 2687, 2246, 5904, 817, 7002, 882, 13, 8603, 975,
	14579, 1172, 198, 220, 979, 432, 28160, 45250, 7002, 11, 54170, 13404, 7329, 11, 476, 73191, 198, 220, 1614, 2687, 2246, 5904, 624, 12, 61242, 1437, 504, 8660, 13, 362, 9155, 11, 48379, 1849, 448, 279, 1852, 22302,
	33327, 264, 8131, 1849, 448, 264, 31773, 4709, 3164, 624, 12, 84468, 4734, 8692, 15491, 7029, 25, 1852, 6888, 16546, 1705, 11, 16745, 7525, 345, 220, 16745, 30611, 11, 16745, 9768, 54175, 973, 11, 323, 902, 501,
	56140, 476, 198, 220, 7982, 31846, 13, 1416, 264, 15491, 2404, 3880, 264, 501, 6783, 11, 9367, 345, 220, 476, 6976, 51, 15792, 46972, 38151, 915, 24335, 311, 975, 11, 2506, 432, 438, 264, 21730, 3080, 279, 16953,
	198, 220, 74449, 429, 897, 382, 565, 12659, 3234, 8082, 7605, 198, 198, 12, 1565, 29454, 21324, 44622, 32135, 323,
}

func TestDenseTrainingLeadership(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	checkpoint := qwenFileSHA256(t, filepath.Join(modelDir, "model.safetensors"))
	config := qwenFileSHA256(t, filepath.Join(modelDir, "config.json"))
	corpus := qwenTokenSHA256(qwenAdaptiveProfileTokens)
	protocolText := fmt.Sprintf(qwenProtocolFormat, len(qwenAdaptiveProfileTokens), qwenLearningRate, qwenMomentum, qwenTrainingSteps)
	protocol := fmt.Sprintf("%x", sha256.Sum256([]byte(protocolText)))
	if checkpoint != qwenCheckpointSHA || config != qwenConfigSHA || corpus != qwenCorpusSHA || protocol != qwenProtocolSHA {
		t.Fatalf("Qwen evidence identity drift: checkpoint=%s config=%s corpus=%s protocol=%s", checkpoint, config, corpus, protocol)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	warm, err := densecausal.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := warm.TrainDeviceResident(worker, [][]int{qwenAdaptiveProfileTokens}, qwenLearningRate, qwenMomentum, densecausal.DeviceTrainingOptions{FrozenLexical: true}); err != nil {
		t.Fatal(err)
	}
	warm = nil
	runtime.GC()
	model, err := densecausal.Load(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	result, err := model.TrainDeviceResident(worker, slices.Repeat([][]int{qwenAdaptiveProfileTokens}, qwenTrainingSteps), qwenLearningRate, qwenMomentum, densecausal.DeviceTrainingOptions{FrozenLexical: true, Measure: true})
	wall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	trajectory, measurement := result.Losses, result.Measurement
	testutil.RequireFiniteDecrease(t, "Qwen trajectory", trajectory, qwenTrainingSteps)
	testutil.RequireClose(t, "Qwen initial loss", trajectory[qwenInitialStep], qwenInitialLoss, qwenInitialLossLimit)
	testutil.RequireRange(t, "Qwen updated loss", trajectory[qwenUpdatedStep], qwenUpdatedLossMinimum, qwenUpdatedLossMaximum)
	if len(measurement.ForwardBackwardSteps) != qwenTrainingSteps || len(measurement.DeviceUpdateSteps) != qwenTrainingSteps {
		t.Fatalf("Qwen phase evidence incomplete: forward=%v update=%v", measurement.ForwardBackwardSteps, measurement.DeviceUpdateSteps)
	}
	cold := measurement.ForwardBackwardSteps[qwenInitialStep] + measurement.DeviceUpdateSteps[qwenInitialStep]
	warmStep := measurement.ForwardBackwardSteps[qwenUpdatedStep] + measurement.DeviceUpdateSteps[qwenUpdatedStep]
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Qwen evidence loss=%v cold=%s warm=%s loop=%s lifecycle=%s peak=%.3fGiB", trajectory, cold, warmStep, measurement.Loop, wall, float64(memory.PeakBytes)/bytesPerGiB)
	if cold >= 2*qwenAdaptiveWarm {
		t.Fatalf("Qwen cold execution %s exceeds twice the adaptive %s warm reference", cold, qwenAdaptiveWarm)
	}
	if warmStep >= qwenAdaptiveWarm {
		t.Fatalf("Qwen warm execution %s does not beat adaptive %s", warmStep, qwenAdaptiveWarm)
	}
	if memory.PeakBytes >= uint64(qwenPeakMemoryGiB*bytesPerGiB) {
		t.Fatalf("Qwen peak %.3fGiB exceeds %dGiB ratchet", float64(memory.PeakBytes)/bytesPerGiB, qwenPeakMemoryGiB)
	}
}

func qwenFileSHA256(t *testing.T, path string) string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func qwenTokenSHA256(tokens []int) string {
	hash := sha256.New()
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], uint32(len(tokens)))
	_, _ = hash.Write(encoded[:])
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(encoded[:], uint32(token))
		_, _ = hash.Write(encoded[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}
