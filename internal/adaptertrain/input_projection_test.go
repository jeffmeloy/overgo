package adaptertrain

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestInputProjectionCheckpoint(t *testing.T) {
	tc := linearCTCOracle(t)[0]
	spec := linearCTCSpec(tc)
	spec.Optimizer.Schedule, spec.Optimizer.Steps = optimizer.ScheduleLinearDecay, 4
	m := testcheck.Must(NewLinearCTC(spec))(t)
	binding := InputProjectionBinding{
		BaseRecipe:      testutil.ArtifactID(t, artifact.KindRecipe, "base"),
		TargetTransform: testutil.ArtifactID(t, artifact.KindProfile, "transform"),
	}
	execution := testcheck.Must(m.Bind(t.Context()))(t)
	example := LinearCTCExample{Hidden: tc.Hidden, Targets: tc.Targets, Frames: tc.Frames}
	forward := testcheck.Must(execution.Select(trainingprogram.PhaseForward))(t)
	testcheck.NoError(t, forward.Run(&example))
	if _, err := m.OptimizerSnapshot(); err == nil {
		t.Fatal("accepted unfinished forward checkpoint")
	}
	if err := m.SaveWeights(t.TempDir(), binding); err == nil {
		t.Fatal("published unfinished checkpoint")
	}
	testcheck.NoError(t, execution.Run(&example))
	state := testcheck.Must(m.OptimizerSnapshot())(t)
	directory := t.TempDir()
	testcheck.NoError(t, m.SaveWeights(directory, binding))
	file := testcheck.Must(os.Open(filepath.Join(directory, trainingprogram.CheckpointWeights)))(t)
	weightsID, _, err := artifact.Identify(artifact.KindTensorSet, file)
	if closeErr := file.Close(); err != nil || closeErr != nil {
		t.Fatalf("hash %v close %v", err, closeErr)
	}
	weights, err := LoadInputProjection(directory, binding, weightsID, tc.Width, m.StorageBytes())
	if err != nil || !slices.Equal(weights, m.WeightSnapshot()) {
		t.Fatalf("weights roundtrip: %v", err)
	}
	resumed := testcheck.Must(NewLinearCTC(spec))(t)
	testcheck.NoError(t, resumed.Restore(weights, state))
	replay := testcheck.Must(resumed.Bind(t.Context()))(t)
	for range 2 {
		replayed := LinearCTCExample{Hidden: tc.Hidden, Targets: tc.Targets, Frames: tc.Frames}
		testcheck.NoError(t, execution.Run(&example))
		testcheck.NoError(t, replay.Run(&replayed))
		left := testcheck.Must(m.OptimizerSnapshot())(t)
		right := testcheck.Must(resumed.OptimizerSnapshot())(t)
		if example.Loss != replayed.Loss || example.Update != replayed.Update || !reflect.DeepEqual(left, right) || !slices.Equal(m.WeightSnapshot(), resumed.WeightSnapshot()) {
			t.Fatal("resumed trajectory differs")
		}
	}
	before := resumed.WeightSnapshot()
	beforeState := testcheck.Must(resumed.OptimizerSnapshot())(t)
	for _, mode := range []string{"weights", "momentum", "configuration", "plan"} {
		t.Run(mode, func(t *testing.T) {
			badWeights := slices.Clone(weights)
			badState := state
			badState.Momentum = slices.Clone(state.Momentum)
			switch mode {
			case "weights":
				badWeights[0] = float32(math.NaN())
			case "momentum":
				badState.Momentum[0] = math.Inf(1)
			case "configuration":
				badState.Config.BaseLearningRate *= 2
			case "plan":
				badState.PlanIdentity = "different"
			}
			if err := resumed.Restore(badWeights, badState); err == nil {
				t.Fatal("invalid restore succeeded")
			}
			afterState, err := resumed.OptimizerSnapshot()
			if err != nil || !reflect.DeepEqual(beforeState, afterState) || !slices.Equal(before, resumed.WeightSnapshot()) {
				t.Fatal("refused restore mutated state")
			}
		})
	}
	for _, mode := range []string{"base", "transform", "digest", "shape", "budget", "missing", "extra", "nonfinite", "placement"} {
		t.Run(mode, func(t *testing.T) {
			candidate := t.TempDir()
			metadata := testcheck.Must(binding.metadata())(t)
			tensors := map[string][]float32{inputProjectionTensor: slices.Clone(weights)}
			shapes := map[string][]int{inputProjectionTensor: {tc.Width, tc.Width}}
			wantBinding, wantID, width, budget := binding, weightsID, tc.Width, m.StorageBytes()
			switch mode {
			case "base":
				wantBinding.BaseRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "wrong")
			case "transform":
				wantBinding.TargetTransform = testutil.ArtifactID(t, artifact.KindProfile, "wrong")
			case "digest":
				wantID = testutil.ArtifactID(t, artifact.KindTensorSet, "wrong")
			case "shape":
				width++
			case "budget":
				budget = uint64(len(weights)*4 - 1)
			case "extra":
				tensors["extra"], shapes["extra"] = []float32{0}, []int{1}
			case "nonfinite":
				tensors[inputProjectionTensor][0] = float32(math.Inf(1))
			case "placement":
				metadata["placement"] = "intermediate-output"
			}
			if mode != "missing" {
				testcheck.NoError(t, safetensors.Save(filepath.Join(candidate, trainingprogram.CheckpointWeights), tensors, shapes, metadata))
			}
			if mode == "nonfinite" {
				file, err := os.Open(filepath.Join(candidate, trainingprogram.CheckpointWeights))
				testcheck.NoError(t, err)
				wantID, _, err = artifact.Identify(artifact.KindTensorSet, file)
				if closeErr := file.Close(); err != nil || closeErr != nil {
					t.Fatalf("hash %v close %v", err, closeErr)
				}
			}
			if _, err := LoadInputProjection(candidate, wantBinding, wantID, width, budget); err == nil {
				t.Fatal("invalid projection admitted")
			}
		})
	}
	t.Log("adapter-only Safetensors roundtrip; two resumed updates exactly match loss, weights, momentum and decaying schedule; unfinished boundaries and nine invalid file/binding cases refused")
}
