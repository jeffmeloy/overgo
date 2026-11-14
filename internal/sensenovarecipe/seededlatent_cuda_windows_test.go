//go:build modeltest && windows

package sensenovarecipe

import (
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/torchrng"
)

func TestSenseNovaSeededLatentMatchesOracle(t *testing.T) {
	cudatest.Require(t)
	var oracle seededLatentOracle
	testcheck.NoError(t, jsonfile.Decode(senseNovaGenerationGold, &oracle))
	if len(oracle.Steps) == 0 {
		t.Fatal("generation oracle has no steps")
	}

	binding := routedlm.SenseNovaBinding()
	cfg, err := routedlm.LoadConfig(senseNovaModelDir, binding)
	testcheck.NoError(t, err)
	flowCfg := testcheck.Must(routedlm.LoadFlowConfig(senseNovaModelDir))(t)
	source := testcheck.Must(safetensors.OpenSource(senseNovaModelDir))(t)
	defer source.Close()
	flowProfile := testcheck.Must(routedlm.InspectFlowProfile(senseNovaModelDir))(t)
	plan := testcheck.Must(routedlm.CompileFlowPlan(source, cfg, flowCfg, flowProfile))(t)
	image := testcheck.Must(plan.ImagePlan(oracle.Request.Width, oracle.Request.Height))(t)

	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	var got []float32
	err = worker.Do(t.Context(), func(state *device.State) error {
		stream := torchrng.NewStream(oracle.Request.Seed)
		defer stream.Close(state)
		got, err = routedlm.SeededFlowLatent(stream, state, plan, image)
		return err
	})
	testcheck.NoError(t, err)

	want := oracle.Steps[0].Z
	wantShape := []int{1, image.Tokens, plan.FlowDim}
	if want.Elements != len(got) || !slices.Equal(want.Shape, wantShape) {
		t.Fatalf("oracle z shape=%v elements=%d, got=%v elements=%d", want.Shape, want.Elements, wantShape, len(got))
	}
	if len(want.Indices) != len(want.Values) || len(want.Indices) == 0 {
		t.Fatalf("bad oracle probes %d/%d", len(want.Indices), len(want.Values))
	}
	for i, rawIndex := range want.Indices {
		index := int(rawIndex)
		if index < 0 || index >= len(got) {
			t.Fatalf("probe %d index %d outside %d", i, index, len(got))
		}
		if got[index] != want.Values[i] {
			t.Fatalf("z probe %d at %d = %.10g, want %.10g", i, index, got[index], want.Values[i])
		}
	}
	t.Logf("SenseNova seeded latent exact: shape=%v probes=%d seed=%d", wantShape, len(want.Indices), oracle.Request.Seed)
}
