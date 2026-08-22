//go:build windows

package latentvideo

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/pytorchzip"
	"overgo/internal/testutil"
)

func TestLiveEditRetainedDenoiser(t *testing.T) {
	cudatest.Require(t)
	repo := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	wanDir := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(filepath.Join(wanDir, "config.json")); os.IsNotExist(err) {
		t.Skipf("UNAVAILABLE: Wan denoiser config: %v", err)
	}
	policy := DenoiserPolicy{
		NumTrainTimesteps: 1000, SinusoidalPeriod: 10000, RotaryFrequencyBase: 10000,
		VAEStride: [3]int{4, 8, 8},
	}
	base, err := LoadDenoiserConfig(wanDir, policy)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := CompileReferenceEditCheckpoint(
		filepath.Join(filepath.Dir(wanDir), "LiveEdit", "ar-forcing_002000.pt"), base,
	)
	if err != nil {
		t.Fatalf("UNAVAILABLE: LiveEdit checkpoint: %v", err)
	}

	source := encodeRetainedDenoiserSource(t, repo, roots.Models)
	current := loadRetainedDenoiserCurrent(t)
	geometry := LatentGeometry{
		Channels: 2 * base.InDim, LatentFrames: 1, LatentHeight: 2, LatentWidth: 4,
		Grid: [3]int{1, 1, 2}, Seq: 2,
	}
	context := readRetainedDenoiserF32(t, testutil.FixturePath(t, "wan", "raw", "g1_cond_context.f32le"), base.TextLen*base.Dim)
	started := time.Now()
	session, err := NewReferenceEditDenoiserCUDASession(t.Context(), checkpoint, geometry, 2, 1, 1, 2, 0, context)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if session != nil {
			_ = session.Close()
		}
	}()
	headE, blockE, err := CompileTimestepConditioning([]float64{900}, session.cold.timestepWeights)
	if err != nil {
		t.Fatal(err)
	}
	var outputs [][]float32
	for frame := range 2 {
		combined := retainedDenoiserChunk(current, source, frame)
		patches, err := session.cold.PatchifyLatent(combined)
		if err != nil {
			t.Fatal(err)
		}
		output, err := session.RunChunk(patches, blockE, headE, frame, true)
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	stats := session.Stats()
	memory, err := session.MemoryStats()
	if err != nil {
		t.Fatal(err)
	}
	execution, err := session.ExecutionStats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.Layers != 2 || stats.Runs != 2 || stats.ContextProjections != 1 || stats.HistoryTokens != geometry.Seq || stats.WeightBytes == 0 {
		t.Fatalf("retained denoiser stats=%+v", stats)
	}
	if session.historyProgram != session.warm[1] || session.historyFrames != 2 {
		t.Fatal("second chunk did not execute through the retained-history graph")
	}
	for _, output := range outputs {
		var energy float64
		for _, value := range output {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("retained denoiser output is non-finite")
			}
			energy += math.Abs(float64(value))
		}
		if energy == 0 {
			t.Fatal("retained denoiser output is empty")
		}
	}
	if testutil.MaxAbsDiff(outputs[0], outputs[1]) == 0 {
		t.Fatal("history-conditioned second chunk reproduced the first chunk")
	}
	retainedWall := time.Since(started)
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	session = nil
	fresh, err := NewReferenceEditDenoiserCUDASession(t.Context(), checkpoint, geometry, 2, 1, 1, 2, 0, context)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	second, err := fresh.cold.PatchifyLatent(retainedDenoiserChunk(current, source, 1))
	if err != nil {
		t.Fatal(err)
	}
	coldSecond, err := fresh.RunChunk(second, blockE, headE, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	historyEffect := testutil.MaxAbsDiff(outputs[1], coldSecond)
	if historyEffect == 0 {
		t.Fatal("retained self-attention history has no numerical effect")
	}
	t.Logf("LiveEdit retained denoiser: layers=%d tokens=%d context_projections=%d runs=%d wall=%.3fs weights=%.3fGiB peak=%.3fGiB graphs=%d launches=%d",
		stats.Layers, stats.HistoryTokens, stats.ContextProjections, stats.Runs, retainedWall.Seconds(),
		float64(stats.WeightBytes)/(1<<30), float64(memory.PeakBytes)/(1<<30), execution.GraphInstantiations, execution.GraphLaunches)
	t.Logf("LiveEdit retained history effect: max_abs=%.6g", historyEffect)
}

func TestLiveEditRetainedDenoiserReplaysExactly(t *testing.T) {
	cudatest.Require(t)
	repo := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	wanDir := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(filepath.Join(wanDir, "config.json")); os.IsNotExist(err) {
		t.Skipf("UNAVAILABLE: Wan denoiser config: %v", err)
	}
	policy := DenoiserPolicy{
		NumTrainTimesteps: 1000, SinusoidalPeriod: 10000, RotaryFrequencyBase: 10000,
		VAEStride: [3]int{4, 8, 8},
	}
	base, err := LoadDenoiserConfig(wanDir, policy)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := CompileReferenceEditCheckpoint(
		filepath.Join(filepath.Dir(wanDir), "LiveEdit", "ar-forcing_002000.pt"), base,
	)
	if err != nil {
		t.Fatal(err)
	}
	geometry := LatentGeometry{
		Channels: checkpoint.Config.InDim, LatentFrames: 3, LatentHeight: 32, LatentWidth: 64,
		Grid: [3]int{3, 16, 32}, Seq: 3 * 16 * 32,
	}
	layers := 2
	context := readRetainedDenoiserF32(t, testutil.FixturePath(t, "wan", "raw", "g1_cond_context.f32le"), base.TextLen*base.Dim)
	session, err := NewReferenceEditDenoiserCUDASession(t.Context(), checkpoint, geometry, layers, 3, 21, 21, 0, context)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	combined := make([]float32, geometry.Channels*geometry.LatentFrames*geometry.LatentHeight*geometry.LatentWidth)
	for index := range combined {
		combined[index] = float32(index%257-128) / 257
	}
	patches, err := session.cold.PatchifyLatent(combined)
	if err != nil {
		t.Fatal(err)
	}
	headE, blockE, err := CompileTimestepConditioning([]float64{900}, session.cold.timestepWeights)
	if err != nil {
		t.Fatal(err)
	}
	run := func() ([]float32, [][2][sha256.Size]byte) {
		t.Helper()
		if err := session.ResetHistory(); err != nil {
			t.Fatal(err)
		}
		output, err := session.RunChunk(patches, blockE, headE, 0, true)
		if err != nil {
			t.Fatal(err)
		}
		digests := make([][2][sha256.Size]byte, layers)
		for layer := range layers {
			for kind, buffer := range []*executor.DeviceBuffer{session.cacheKeys[layer], session.cacheValues[layer]} {
				value, err := buffer.Value(session.cold.currentSelfKeys[layer].Shape)
				if kind == 1 {
					value, err = buffer.Value(session.cold.currentSelfVals[layer].Shape)
				}
				if err != nil {
					t.Fatal(err)
				}
				elements, err := value.Shape.Elements()
				if err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, int(elements)*4)
				if err := session.worker.Do(t.Context(), func(state *device.State) error {
					return state.Driver.MemcpyDtoH(raw, value.Pointer)
				}); err != nil {
					t.Fatal(err)
				}
				digests[layer][kind] = sha256.Sum256(raw)
			}
		}
		return output, digests
	}
	first, firstDigests := run()
	second, secondDigests := run()
	for layer := range firstDigests {
		for kind, name := range []string{"key", "value"} {
			if firstDigests[layer][kind] != secondDigests[layer][kind] {
				t.Fatalf("retained replay layer=%d %s digest differs", layer, name)
			}
		}
	}
	if worst := testutil.MaxAbsDiff(first, second); worst != 0 {
		t.Fatalf("retained replay head max_abs=%.6g", worst)
	}
}

func encodeRetainedDenoiserSource(t testing.TB, repo, models string) []float32 {
	t.Helper()
	checkpoint := filepath.Join(models, "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := CompileVAEEncoderPlan(catalog.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	profile := SourceCodecProfile{InputChannels: graph.InputChannels, LatentChannels: graph.LatentChannels, Stride: graph.Stride, LatentStats: loadSourceCodecStats(t)}
	plan, err := CompileSourceCodecBoundary(profile, SourceVideoShape{Channels: 3, Frames: 5, Height: 32, Width: 32})
	if err != nil {
		t.Fatal(err)
	}
	input := loadRealSourceCrops(t, filepath.Join(repo, "build", "latentvideo", "current_prod50", "frame_00.f32le"), plan.Source)
	session, err := NewVAEEncoderCUDASession(checkpoint, graph, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	latent, _, err := session.Encode(t.Context(), plan, input)
	if err != nil {
		t.Fatal(err)
	}
	return latent
}

func loadRetainedDenoiserCurrent(t testing.TB) []float32 {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "wan", "g4_denoise.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Final struct {
			Values []float32 `json:"values"`
		} `json:"final_latent"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Final.Values) != 16*2*2*4 {
		t.Fatalf("Wan retained latent elements=%d", len(manifest.Final.Values))
	}
	return manifest.Final.Values
}

func retainedDenoiserChunk(current, source []float32, frame int) []float32 {
	const channels, frames, height, width = 16, 2, 2, 4
	const sourceHeight, sourceWidth = 4, 4
	spatial := height * width
	chunk := make([]float32, 2*channels*spatial)
	for channel := range channels {
		copy(chunk[channel*spatial:(channel+1)*spatial], current[(channel*frames+frame)*spatial:])
		for y := range height {
			src := ((channel*frames+frame)*sourceHeight + y) * sourceWidth
			dst := (channels+channel)*spatial + y*width
			copy(chunk[dst:dst+width], source[src:src+width])
		}
	}
	return chunk
}

func readRetainedDenoiserF32(t testing.TB, path string, elements int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != elements*4 {
		t.Fatalf("retained context bytes=%d want=%d", len(raw), elements*4)
	}
	values := make([]float32, elements)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
	}
	return values
}
