//go:build windows

package sensenovarecipe

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/torchrng"
)

const senseNovaGenerationGold = `..\..\fixtures\sensenova\generation_native_256.json`

// senseNovaModelDir is the real SenseNova artifact under the model roots,
// resolved once from the repository root two directories up: a value, not a
// function of a test, because the pinned production-oracle test reads it as
// one.
var senseNovaModelDir = func() string {
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		panic(err)
	}
	return roots.ResolveModelPath("SenseNova-U1-8B-MoT-Infographic-V3")
}()

type seededLatentOracle struct {
	Request struct {
		Width  int   `json:"width"`
		Height int   `json:"height"`
		Seed   int64 `json:"seed"`
	} `json:"request"`
	Steps []struct {
		Z struct {
			Shape    []int     `json:"shape"`
			Elements int       `json:"elements"`
			Indices  []int64   `json:"indices"`
			Values   []float32 `json:"values"`
		} `json:"z"`
	} `json:"steps"`
}

func TestSenseNovaSeededLatentMatchesOracle(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_SENSENOVA_BASELINE") != "1" {
		testskip.NotApplicable(t, "set OVERGO_SENSENOVA_BASELINE=1 for SenseNova generation evidence")
	}
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
