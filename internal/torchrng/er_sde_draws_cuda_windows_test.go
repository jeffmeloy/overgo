//go:build windows

package torchrng

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// erSDEDraws is each fresh noise draw ComfyUI's er_sde made, torch.randn from one CUDA generator seeded with the
// request's seed (scripts/python_golden_references/comfy_er_sde_golden.py, torch 2.14 on this card): a small latent's
// whole, and MOSS's latent at 1024 x 576 by SHA-256 and first values.
type erSDEDraws struct {
	Seed  int64 `json:"seed"`
	Cases []struct {
		Name       string      `json:"name"`
		Shape      []int       `json:"shape"`
		Draws      int         `json:"draws"`
		DrawSHA256 []string    `json:"draw_sha256"`
		DrawHead   [][]float32 `json:"draw_head"`
		Noise      [][]float32 `json:"noise"`
	} `json:"cases"`
}

// TestERSDEDrawsNoiseAsTorchDoes holds ER-SDE's fresh noise (moss-ultra-sections/er-sde): one Stream seeded with the
// request's seed gives, fill after fill, every draw ComfyUI's er_sde took from torch's CUDA generator over eight steps,
// bit for bit, at a small latent and at MOSS's [1, 128, 36, 64].
func TestERSDEDrawsNoiseAsTorchDoes(t *testing.T) {
	cudatest.Require(t)
	var golden erSDEDraws
	testcheck.NoError(t, jsonfile.Decode("testdata/comfy_er_sde_draws.json", &golden))
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	moss := false
	for _, c := range golden.Cases {
		elements := 1
		for _, extent := range c.Shape {
			elements *= extent
		}
		draws := make([][]float32, c.Draws)
		testcheck.NoError(t, worker.Do(t.Context(), func(state *device.State) error {
			stream := NewStream(golden.Seed)
			defer stream.Close(state)
			for i := range draws {
				draws[i] = make([]float32, elements)
				if err := stream.FillHost(state, draws[i]); err != nil {
					return err
				}
			}
			return nil
		}))
		for i, draw := range draws {
			digest := sha256.Sum256(driver.Bytes(draw))
			if hex.EncodeToString(digest[:]) != c.DrawSHA256[i] || !slices.Equal(draw[:len(c.DrawHead[i])], c.DrawHead[i]) ||
				c.Noise != nil && !slices.Equal(draw, c.Noise[i]) {
				t.Errorf("%s draw %d of %d: %v…, torch's %v…", c.Name, i+1, c.Draws, draw[:len(c.DrawHead[i])], c.DrawHead[i])
			}
		}
		moss = moss || slices.Equal(c.Shape, []int{1, 128, 36, 64}) && c.Draws == 7
	}
	if !moss {
		t.Error("the golden holds no seven draws at MOSS's latent")
	}
}
