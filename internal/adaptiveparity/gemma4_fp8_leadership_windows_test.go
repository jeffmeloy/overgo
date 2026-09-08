//go:build windows

package adaptiveparity

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/strictjson"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type gemma4FP8PerformanceEvidence struct {
	Schema                  string `json:"schema"`
	SourceCommit            string `json:"source_commit"`
	SourceEntrypoint        string `json:"source_entrypoint"`
	Protocol                string `json:"protocol"`
	GoldenSHA256            string `json:"golden_sha256"`
	CandidateArtifactSHA256 string `json:"candidate_artifact_sha256"`
	SourceConfigSHA256      string `json:"source_config_sha256"`
	SourceIndexSHA256       string `json:"source_index_sha256"`
	SourceShards            []struct {
		Name   string `json:"name"`
		Bytes  uint64 `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"source_shards"`
	Environment struct {
		GPU    string `json:"gpu"`
		Driver string `json:"driver"`
		Go     string `json:"go"`
		OSArch string `json:"os_arch"`
	} `json:"environment"`
	Prompt    string              `json:"prompt"`
	InputIDs  []tokenizer.TokenID `json:"input_ids"`
	OutputIDs []tokenizer.TokenID `json:"output_ids"`
	Runs      []struct {
		LoadNanos       uint64 `json:"load_nanos"`
		GenerationNanos uint64 `json:"generation_nanos"`
		WallNanos       uint64 `json:"wall_nanos"`
		PeakOwnedBytes  uint64 `json:"peak_owned_bytes"`
	} `json:"runs"`
}

func TestGemma4FP8Leadership(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_GEMMA4_BASELINE") != "1" {
		t.Skip("set OVERGO_GEMMA4_BASELINE=1 for Gemma4 FP8 leadership")
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	root := testutil.RepoRoot(t)
	var evidence gemma4FP8PerformanceEvidence
	evidencePath := filepath.Join(root, "fixtures", "adaptive_gemma4_fp8_performance.json")
	evidenceBytes, err := os.ReadFile(evidencePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := strictjson.DecodeBytes(evidenceBytes, &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Schema != "overgo/adaptive-gemma4-fp8-performance/v1" || evidence.SourceCommit == "" ||
		evidence.SourceEntrypoint != "go/extmodel/alternating_window_decoder_cuda_windows_test.go:TestGemma4Device_GoldenGate" ||
		evidence.Protocol != "case=0;greedy=true;max_new_tokens=16;wall=generation;candidate_cache_page_tokens=32;reference_cache_tokens=256;peak=runtime_owned_cuda" ||
		evidence.GoldenSHA256 != "ccb91b5e6912c23805b2253d16c193fac8445396444f3f0fc4d370d39eaeaa94" ||
		evidence.SourceConfigSHA256 != "18bd83445d43b1c5ffb31e81ddf0f05880bcc2d7a9220967ccab4866240fdc59" ||
		evidence.SourceIndexSHA256 != "8bec4873473c10d187ad0c333945f44a2e7a4103dace4f1a344d25adbd76647f" ||
		len(evidence.SourceShards) != 2 || evidence.SourceShards[0].Name != "model-00001-of-00002.safetensors" ||
		evidence.SourceShards[0].Bytes != 9969043746 || evidence.SourceShards[0].SHA256 != "64088516d8c65da90319b2fad0a8df850de8e8ceaeb192a95cc62e9da3223742" ||
		evidence.SourceShards[1].Name != "model-00002-of-00002.safetensors" || evidence.SourceShards[1].Bytes != 5463692670 ||
		evidence.SourceShards[1].SHA256 != "72358dc742bbe9851c05b37f4c7b286fb5585034f2c342915456ac229bd6c5a3" ||
		evidence.Environment.GPU == "" || evidence.Environment.Driver == "" || evidence.Environment.Go != runtime.Version() ||
		evidence.Environment.OSArch != runtime.GOOS+"/"+runtime.GOARCH || len(evidence.InputIDs) == 0 || len(evidence.OutputIDs) == 0 || len(evidence.Runs) < 5 {
		t.Fatal("adaptive Gemma4 FP8 evidence fingerprint differs")
	}
	referenceWalls := make([]uint64, len(evidence.Runs))
	referencePeak := uint64(math.MaxUint64)
	for index, run := range evidence.Runs {
		if run.LoadNanos == 0 || run.GenerationNanos == 0 || run.WallNanos < run.LoadNanos+run.GenerationNanos || run.PeakOwnedBytes == 0 {
			t.Fatal("adaptive Gemma4 FP8 evidence run invalid")
		}
		referenceWalls[index] = run.GenerationNanos
		referencePeak = min(referencePeak, run.PeakOwnedBytes)
	}
	slices.Sort(referenceWalls)
	referenceWall := referenceWalls[len(referenceWalls)/2]
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-12B-it-fp8-native.gguf")
	assertSHA256(t, modelPath, evidence.CandidateArtifactSHA256)
	started := time.Now()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(t.Context(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	loadWall := time.Since(started)
	loadMemory, err := runner.DeviceMemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	// The reference wall is the MEDIAN of its five recorded runs, so the
	// candidate measures the same statistic: three greedy generations,
	// each token-exact, compared by median -- one cold-clock or
	// contended run cannot decide the claim in either direction.
	var generated []tokenizer.TokenID
	var memory driver.MemoryStats
	generationWalls := make([]uint64, 0, 3)
	for run := range 3 {
		generationStarted := time.Now()
		ids, _, err := runner.Generate(t.Context(), evidence.Prompt, inference.GenerateOptions{
			MaxNewTokens:   16,
			Sampler:        greedy,
			DeviceGreedy:   true,
			PromptTokenIDs: evidence.InputIDs,
		})
		generationWalls = append(generationWalls, uint64(time.Since(generationStarted)))
		if err != nil {
			t.Fatal(err)
		}
		generated = ids[len(evidence.InputIDs):]
		if !slices.Equal(generated, evidence.OutputIDs) {
			t.Fatalf("Gemma4 run %d generated IDs %v, want %v", run, generated, evidence.OutputIDs)
		}
		if run == 0 {
			// The reference peak is the MINIMUM across its runs -- its
			// cleanest single generation. The candidate matches that
			// statistic with its own first-generation peak; later runs
			// re-stage a prefill on top of run one's retained decode
			// residency, a coexistence the reference minimum never holds.
			if memory, err = runner.DeviceMemoryStats(t.Context()); err != nil {
				t.Fatal(err)
			}
		}
	}
	slices.Sort(generationWalls)
	generationWall := time.Duration(generationWalls[len(generationWalls)/2])
	wall := time.Since(started)
	t.Logf("Gemma4 FP8 leadership load=%s generation=%s wall=%s load_current=%d load_peak=%d current=%d peak=%d peak_allocation=%d largest_live=%d reference_generation=%s reference_peak=%d over=%d output=%v", loadWall, generationWall, wall, loadMemory.CurrentBytes, loadMemory.PeakBytes, memory.CurrentBytes, memory.PeakBytes, memory.PeakAllocationBytes, memory.LargestLiveBytes, time.Duration(referenceWall), referencePeak, int64(memory.PeakBytes)-int64(referencePeak), generated)
	// The peak-time ledger is the evidence for WHERE the bytes live; the
	// load/end diff separates weights-residency classes from what
	// generation allocated and kept.
	excess := int64(memory.PeakBytes) - int64(referencePeak)
	for index, class := range memory.PeakLedger {
		if index < 12 || int64(class.Bytes) <= excess {
			t.Logf("peak ledger[%d]: bytes=%d count=%d total=%d", index, class.Bytes, class.Count, class.Bytes*class.Count)
		}
	}
	t.Logf("peak ledger classes=%d", len(memory.PeakLedger))
	loadCounts := map[uint64]uint64{}
	for _, class := range loadMemory.PeakLedger {
		loadCounts[class.Bytes] = class.Count
	}
	for _, class := range memory.PeakLedger {
		if grown := class.Count - loadCounts[class.Bytes]; grown > 0 {
			t.Logf("generation-time class: bytes=%d grew=%d total=%d", class.Bytes, grown, class.Bytes*grown)
		}
	}
	if uint64(generationWall) > referenceWall {
		t.Fatalf("Gemma4 generation %s exceeds adaptive %s", generationWall, time.Duration(referenceWall))
	}
	if memory.PeakBytes > referencePeak {
		t.Fatalf("Gemma4 peak %d exceeds adaptive %d", memory.PeakBytes, referencePeak)
	}
}
