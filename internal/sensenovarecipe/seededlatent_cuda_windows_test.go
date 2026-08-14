//go:build windows

package sensenovarecipe

import (
	"context"
	"os"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/torchrng"
)

const (
	senseNovaModelDir       = `C:\Users\jeffm\adaptive_new\models\SenseNova-U1-8B-MoT-Infographic-V3`
	senseNovaGenerationGold = `..\..\fixtures\sensenova\generation_native_256.json`
)

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
		t.Skip("set OVERGO_SENSENOVA_BASELINE=1 for SenseNova generation evidence")
	}
	var oracle seededLatentOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &oracle); err != nil {
		t.Fatal(err)
	}
	if len(oracle.Steps) == 0 {
		t.Fatal("generation oracle has no steps")
	}

	binding := routedlm.SenseNovaBinding()
	cfg, err := routedlm.LoadConfig(senseNovaModelDir, binding)
	if err != nil {
		t.Fatal(err)
	}
	flowCfg, err := routedlm.LoadFlowConfig(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	plan, err := routedlm.CompileFlowPlan(source, cfg, flowCfg, routedlm.SenseNovaFlowBinding())
	if err != nil {
		t.Fatal(err)
	}
	image, err := plan.ImagePlan(oracle.Request.Width, oracle.Request.Height)
	if err != nil {
		t.Fatal(err)
	}

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var got []float32
	err = worker.Do(context.Background(), func(state *device.State) error {
		stream := torchrng.NewStream(oracle.Request.Seed)
		defer stream.Close(state)
		got, err = routedlm.SeededFlowLatent(stream, state, plan, image)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}

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
