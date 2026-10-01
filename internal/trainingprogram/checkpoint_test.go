package trainingprogram

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
)

// publishTestCheckpoint publishes spec with stand-in weights bytes into a
// fresh directory.
func publishTestCheckpoint(t *testing.T, spec CheckpointSpec) (Checkpoint, string) {
	t.Helper()
	target := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint, err := PublishCheckpoint(target, spec, func(stage string) error {
		return os.WriteFile(filepath.Join(stage, CheckpointWeights), []byte("weights"), 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return checkpoint, target
}

func checkpointTestSpec(t *testing.T, state hostoptimizer.State) CheckpointSpec {
	t.Helper()
	id := func(kind artifact.Kind, value string) artifact.ID {
		result, err := artifact.IdentifyBytes(kind, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	model := id(artifact.KindModel, "model")
	return CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run"), Program: id(artifact.KindRecipe, "program"), Model: model,
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Stream:    DatasetState{Identity: id(artifact.KindDatasetShard, "stream"), Position: 7},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG: []RNGState{
			{Name: "data", Algorithm: id(artifact.KindProfile, "rng"), Seed: 1, Counter: 7},
			{Name: "augmentation", Algorithm: id(artifact.KindProfile, "rng"), Seed: 2, Counter: 3},
		},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Projectors: []artifact.ID{id(artifact.KindProjector, "projector")},
		Codecs:     []artifact.ID{id(artifact.KindProfile, "codec")},
		Lineage:    []LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}
}

func TestProductionCheckpointResumeExact(t *testing.T) {
	plan, err := hostoptimizer.CompilePlan(2, []hostoptimizer.GroupSpec{{Name: "weight", Start: 0, End: 2, Rows: 1, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	muon, err := hostoptimizer.New([]float32{1, 2}, []float32{3, 4}, plan, hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	muon.Step()
	spec := checkpointTestSpec(t, muon.Snapshot())
	written, target := publishTestCheckpoint(t, spec)
	loaded, err := LoadCheckpoint(target)
	if err != nil || loaded.ID() != written.ID() || loaded.Stream != spec.Stream || loaded.Optimizer.Step != 1 || len(loaded.ArtifactLineage()) != 1 {
		t.Fatalf("round trip = (%+v, %v)", loaded, err)
	}
	if _, err := PublishCheckpoint(target, spec, func(string) error { return nil }); err == nil {
		t.Fatal("checkpoint overwrite accepted")
	}
	if err := os.WriteFile(filepath.Join(target, CheckpointWeights), []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(target); err == nil {
		t.Fatal("modified weights accepted")
	}
	failed := filepath.Join(filepath.Dir(target), "failed")
	if _, err := PublishCheckpoint(failed, spec, func(string) error { return errors.New("stop") }); err == nil {
		t.Fatal("failed publication accepted")
	}
	if _, err := os.Stat(failed); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed target exists: %v", err)
	}
	spec.Accumulation = 1
	if _, err := PublishCheckpoint(filepath.Join(t.TempDir(), "accumulating"), spec, func(stage string) error {
		return os.WriteFile(filepath.Join(stage, CheckpointWeights), []byte("weights"), 0o644)
	}); err == nil {
		t.Fatal("mid-accumulation checkpoint accepted")
	}
}

// TestResumeSidecarRoundTripsBitExact holds the resume sidecar to the
// trainer's own state: the checkpoint document carries no momentum, stream
// or RNG; the momentum file holds the trainer's dtype, four bytes a value
// for a device stepper's f32 and eight for the host optimizer's f64; and
// loading returns every momentum bit, the step, config, plan, stream and RNG
// the trainer published. A momentum file or sidecar that differs from the
// digests the document chain names, and a version 1 document that inlined
// its momentum, refuse.
func TestResumeSidecarRoundTripsBitExact(t *testing.T) {
	plan, err := hostoptimizer.CompilePlan(6, []hostoptimizer.GroupSpec{{Name: "weight", End: 6, Rows: 2, Cols: 3}})
	if err != nil {
		t.Fatal(err)
	}
	config := hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9}
	extremes32 := []float32{math.Float32frombits(1), -0.0, math.MaxFloat32, -math.SmallestNonzeroFloat32, 1.0 / 3, -7.25}
	extremes64 := []float64{math.Float64frombits(1), math.Copysign(0, -1), math.MaxFloat64, 1.0 / 3, -1e-300, 0.1}
	for _, state := range []struct {
		name  string
		width int
		state hostoptimizer.State
	}{
		{"device-f32", 4, hostoptimizer.State{PlanIdentity: plan.Identity(), Config: config, Step: 3, MomentumF32: extremes32}},
		{"host-f64", 8, hostoptimizer.State{PlanIdentity: plan.Identity(), Config: config, Step: 3, Momentum: extremes64}},
	} {
		t.Run(state.name, func(t *testing.T) {
			spec := checkpointTestSpec(t, state.state)
			written, target := publishTestCheckpoint(t, spec)
			document, err := os.ReadFile(filepath.Join(target, CheckpointFilename))
			if err != nil {
				t.Fatal(err)
			}
			for _, inlined := range []string{`"momentum"`, `"stream"`, `"rng"`, `"optimizer"`} {
				if bytes.Contains(document, []byte(inlined)) {
					t.Fatalf("checkpoint document inlines %s: %s", inlined, document)
				}
			}
			if id, err := artifact.IdentifyBytes(artifact.KindCheckpoint, document); err != nil || id != written.ID() {
				t.Fatalf("checkpoint file digest = %s (%v), want the checkpoint identity %s", id, err, written.ID())
			}
			momentumPath := filepath.Join(target, momentumFilename)
			if info, err := os.Stat(momentumPath); err != nil || info.Size() != int64(state.width*spec.ParameterCount) {
				t.Fatalf("momentum file = (%v, %v), want %d bytes a value", info, err, state.width)
			}
			loaded, err := LoadCheckpoint(target)
			if err != nil {
				t.Fatal(err)
			}
			got := loaded.Optimizer
			if loaded.ID() != written.ID() || loaded.Resume != written.Resume || got.Step != state.state.Step ||
				got.Config != state.state.Config || got.PlanIdentity != state.state.PlanIdentity ||
				loaded.Stream != spec.Stream || !slices.Equal(loaded.RNG, written.RNG) ||
				!slices.EqualFunc(got.MomentumF32, state.state.MomentumF32, func(left, right float32) bool { return math.Float32bits(left) == math.Float32bits(right) }) ||
				!slices.EqualFunc(got.Momentum, state.state.Momentum, func(left, right float64) bool { return math.Float64bits(left) == math.Float64bits(right) }) {
				t.Fatalf("loaded resume state %+v differs from the published %+v", got, state.state)
			}
			original, err := os.ReadFile(momentumPath)
			if err != nil {
				t.Fatal(err)
			}
			flipped := slices.Clone(original)
			flipped[0] ^= 1
			if err := os.WriteFile(momentumPath, flipped, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadCheckpoint(target); err == nil || !strings.Contains(err.Error(), "momentum file differs") {
				t.Fatalf("a changed momentum bit loaded: %v", err)
			}
			if err := os.WriteFile(momentumPath, original[:len(original)-1], 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadCheckpoint(target); err == nil {
				t.Fatal("a truncated momentum file loaded")
			}
		})
	}
	legacy := []byte(`{"version":1,"optimizer":{"momentum":[0.5]},"stream":{},"rng":[]}`)
	if _, err := ParseCheckpoint(legacy); err == nil || !strings.Contains(err.Error(), "resume sidecar") {
		t.Fatalf("a version 1 document with inline momentum = %v, want a refusal naming the resume sidecar", err)
	}
}
