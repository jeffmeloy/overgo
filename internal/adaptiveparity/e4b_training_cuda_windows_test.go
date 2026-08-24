//go:build windows

package adaptiveparity_test

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/projector"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

const (
	e4bModelSHA      = "cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501"
	e4bProjectorSHA  = "185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e"
	e4bVisionDataSHA = "6d30559c2d237385a19b1ff68817c25c5e999f3d44290cf029676b031b80b710"
	e4bPositionSHA   = "f6d78057fb188166f763d174659f30abfea1720005b9300e13e49a3e9d133089"
	e4bAudioDataSHA  = "6faf97d1bf73ab3631f38332bf0539d43ebd0eddaf76865728288032dc939fca"
	e4bAudioRopeBase = float32(10000) // Adaptive fixture profile fact.
)

func TestGemmaE4BRealTraining(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	visionPath := filepath.Join(roots.Models, "..", "fixtures", "e4b_vision", "pixel_values.f32")
	positionPath := filepath.Join(roots.Models, "..", "fixtures", "e4b_vision", "positions.i32")
	audioPath := filepath.Join(roots.Models, "..", "fixtures", "e4b_audio", "input_features.f32")
	for path, want := range map[string]string{
		modelPath: e4bModelSHA, projectorPath: e4bProjectorSHA, visionPath: e4bVisionDataSHA,
		positionPath: e4bPositionSHA, audioPath: e4bAudioDataSHA,
	} {
		if got := qwenFileSHA256(t, filepath.Clean(path)); got != want {
			t.Fatalf("E4B input %s identity=%s, want %s", filepath.Base(path), got, want)
		}
	}

	ctx := context.Background()
	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](ctx, projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	pixels := readE4BF32(t, visionPath)
	positions := readE4BI32(t, positionPath)
	vision, err := runner.EncodeVisionPatches(ctx, pixels, positions)
	if err != nil {
		t.Fatal(err)
	}
	audioProfile, err := projector.NewAudioProjectionProfile(e4bAudioRopeBase)
	if err != nil {
		t.Fatal(err)
	}
	audio, err := runner.EncodeAudioFeatures(ctx, readE4BF32(t, audioPath), 39, audioProfile)
	if err != nil {
		t.Fatal(err)
	}

	loaded := time.Now()
	continuous, topology, err := adaptertrain.LoadArtifact(ctx, modelPath, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	loadWall := time.Since(loaded)
	if !continuous.Program().ID().Valid() || continuous.ParameterCount() == 0 ||
		!topology.ProjectedInput().PerLayerEmbeddings || topology.Forward().AlternateStates() {
		t.Fatal("E4B compiled training authority differs from the artifact")
	}
	sliding, err := topology.Layer(0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := topology.Layer(5)
	if err != nil {
		t.Fatal(err)
	}
	shared, err := topology.Layer(24)
	if err != nil {
		t.Fatal(err)
	}
	if !sliding.Sliding || full.Sliding || !shared.SharedKV || shared.HasKV || shared.KVSource >= shared.Layer {
		t.Fatalf("E4B topology: sliding=%+v full=%+v shared=%+v", sliding, full, shared)
	}

	const sampleRows = 2
	hidden := len(vision.Embeddings.Data) / vision.SoftTokens
	if hidden <= 0 || len(audio.Embeddings.Data)/audio.SoftTokens != hidden {
		t.Fatal("E4B projector outputs have inconsistent language width")
	}
	examples := make([]adaptertrain.Example, 3)
	examples[0], err = continuous.BuildExample(ctx, modelPath, nil, []uint32{2, 105}, []uint32{105, 2364}, "text")
	if err != nil {
		t.Fatal(err)
	}
	examples[1], err = continuous.BuildExample(ctx, modelPath,
		vision.Embeddings.Data[:sampleRows*hidden], slices.Repeat([]uint32{258880}, sampleRows),
		[]uint32{258880, 258882}, "image")
	if err != nil {
		t.Fatal(err)
	}
	examples[2], err = continuous.BuildExample(ctx, modelPath,
		audio.Embeddings.Data[:sampleRows*hidden], slices.Repeat([]uint32{258881}, sampleRows),
		[]uint32{258881, 258883}, "audio")
	if err != nil {
		t.Fatal(err)
	}

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	started := time.Now()
	trajectory := make([]float64, len(examples))
	for index, example := range examples {
		trajectory[index], err = continuous.Step(worker, example)
		if err != nil || math.IsNaN(trajectory[index]) || math.IsInf(trajectory[index], 0) {
			t.Fatalf("E4B %s step loss=%g err=%v", example.Kind, trajectory[index], err)
		}
	}
	trainWall := time.Since(started)

	partial, _, err := adaptertrain.LoadArtifact(ctx, modelPath, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := partial.Step(worker, examples[0]); err != nil {
		t.Fatal(err)
	}
	checkpointDir := filepath.Join(t.TempDir(), "e4b-adapter")
	checkpointSpec := e4bCheckpointSpec(t, partial, e4bModelSHA+e4bProjectorSHA)
	written, err := trainingprogram.PublishCheckpoint(checkpointDir, checkpointSpec, func(stage string) error {
		return partial.SaveWeights(adaptertrain.CheckpointWeightsPath(stage))
	})
	if err != nil {
		t.Fatal(err)
	}
	resumed, _, err := adaptertrain.LoadArtifact(ctx, modelPath, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	loadedCheckpoint, err := trainingprogram.LoadCheckpoint(checkpointDir)
	if err != nil || loadedCheckpoint.ID() != written.ID() {
		t.Fatalf("E4B checkpoint load=%s err=%v", loadedCheckpoint.ID(), err)
	}
	if err := resumed.RestoreWeights(checkpointDir); err != nil {
		t.Fatal(err)
	}
	if err := resumed.Restore(adaptertrain.State{Optimizer: loadedCheckpoint.Optimizer}); err != nil {
		t.Fatal(err)
	}
	for _, example := range examples[1:] {
		if _, err := resumed.Step(worker, example); err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(continuous.WeightSnapshot(), resumed.WeightSnapshot()) ||
		!reflect.DeepEqual(continuous.Snapshot(), resumed.Snapshot()) {
		t.Fatal("E4B checkpoint resume differs from uninterrupted Muon trajectory")
	}

	t.Logf("Gemma E4B layer 0 adapter: params=%d losses(text,image,audio)=%v load=%s train=%s program=%s checkpoint=%s",
		continuous.ParameterCount(), trajectory, loadWall, trainWall, continuous.Program().ID(), written.ID())
	t.Logf("topology: layer0=sliding layer5=full layer24=shared-kv(source=%d); artifact declares no AltUp or Laurel", shared.KVSource)
}

func readE4BF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil || len(raw)%4 != 0 {
		t.Fatalf("read F32 %s: bytes=%d err=%v", path, len(raw), err)
	}
	result := make([]float32, len(raw)/4)
	for index := range result {
		result[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return result
}

func readE4BI32(t *testing.T, path string) []int32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil || len(raw)%4 != 0 {
		t.Fatalf("read I32 %s: bytes=%d err=%v", path, len(raw), err)
	}
	result := make([]int32, len(raw)/4)
	for index := range result {
		result[index] = int32(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return result
}

func e4bCheckpointSpec(t *testing.T, trained *adaptertrain.Model, evidence string) trainingprogram.CheckpointSpec {
	t.Helper()
	id := func(kind artifact.Kind, value string) artifact.ID {
		result, err := artifact.IdentifyBytes(kind, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	modelID := id(artifact.KindModel, e4bModelSHA)
	dataset := id(artifact.KindDataset, evidence)
	split := id(artifact.KindDatasetShard, evidence+"/split")
	return trainingprogram.CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "e4b-real-adapter-run"), Program: trained.Program().ID(), Model: modelID,
		Dataset: dataset, Split: split, Stream: trainingprogram.DatasetState{Identity: split, Position: 1},
		Optimizer: trained.Snapshot().Optimizer, ParameterCount: trained.ParameterCount(),
		RNG: []trainingprogram.RNGState{
			{Name: "augmentation", Algorithm: id(artifact.KindProfile, "none"), Seed: 1},
			{Name: "data", Algorithm: id(artifact.KindProfile, "ordered"), Seed: 1, Counter: 1},
		},
		Processors: []artifact.ID{id(artifact.KindProfile, "e4b-fixture-processors")},
		Projectors: []artifact.ID{id(artifact.KindProjector, e4bProjectorSHA)},
		Lineage: []trainingprogram.LineageParent{
			{Artifact: modelID, Relation: artifact.RelationTrainedFrom},
			{Artifact: dataset, Relation: artifact.RelationDerivedFrom},
		},
	}
}
