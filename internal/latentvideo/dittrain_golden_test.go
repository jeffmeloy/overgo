//go:build integration

package latentvideo

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestDiTTrainerGoldenParityG3: the trainer's f32 host mirror forward vs the
// committed exact CUDA capture — the g2 timestep conditioning through the
// trainer's own trainable-tensor path (bit parity, tolerance 0), then the
// full patch-embed -> 30-block -> head forward on the real g3 step-0 sample
// with the committed g1 projected context: block-0 intra-block seams, every
// block output, and the unpatchified branch output, on both branches.
func TestDiTTrainerGoldenParityG3(t *testing.T) {
	if testing.Short() {
		t.Skip("integration excluded by -short: loads the full 5.7GB trainable tensor set")
	}
	manifest, dir := loadDenoiseManifest(t, "g3_denoise.json")
	modelDir := wanModelDir(t)
	config, err := LoadDenoiserConfig(modelDir, referenceDenoiserPolicy)
	if err != nil {
		t.Fatal(err)
	}
	tensors, textDim, err := LoadDiTTrainerTensors(modelDir, config)
	if err != nil {
		t.Fatal(err)
	}
	geometry, err := config.CompileLatentGeometry(manifest.Request.Frames, manifest.Request.Width, manifest.Request.Height)
	if err != nil {
		t.Fatal(err)
	}
	shape := manifest.LatentShape
	if geometry.Channels != shape.Channels || geometry.LatentFrames != shape.LatentFrames ||
		geometry.LatentHeight != shape.LatentHeight || geometry.LatentWidth != shape.LatentWidth ||
		geometry.Seq != shape.SeqLen {
		t.Fatalf("latent geometry %+v differs from golden %+v", geometry, shape)
	}
	trainer, err := NewDiTTrainer(config, textDim, geometry, tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	t.Logf("trainable_parameters=%d text_dim=%d", trainer.ParameterCount(), textDim)

	// g2 timestep conditioning through the trainer's trainable path: same
	// engine class as the capture, so tolerance is 0 (bit parity).
	raw, err := os.ReadFile(filepath.Join(dir, "g2_timestep_conditioning.json"))
	if err != nil {
		t.Fatalf("missing required golden manifest: %v", err)
	}
	var g2 struct {
		HeadE    g1Tensor `json:"head_e"`
		BlockE   g1Tensor `json:"block_e"`
		Schedule struct {
			Timesteps []int64 `json:"timesteps"`
		} `json:"schedule"`
	}
	if err := json.Unmarshal(raw, &g2); err != nil {
		t.Fatal(err)
	}
	headEAll := loadG1Tensor(t, dir, g2.HeadE)   // [steps, dim]
	blockEAll := loadG1Tensor(t, dir, g2.BlockE) // [steps, 6, dim]
	d := config.Dim
	timeActs, err := trainer.timestepConditioning(float64(g2.Schedule.Timesteps[0]))
	if err != nil {
		t.Fatal(err)
	}
	requireParity(t, "g2 head_e[0] via trainer", timeActs.headE, headEAll[:d], 0)
	requireParity(t, "g2 block_e[0] via trainer", timeActs.blockE, blockEAll[:6*d], 0)

	// Step-0 forward on the captured sample with the committed projected
	// contexts (the REAL g1 T5-derived conditioning rows).
	contextElements := config.TextLen * config.Dim
	sampleIn := manifest.trace(t, dir, "step_00_sample_in")
	branches := []struct {
		name    string
		context []float32
	}{
		{"cond", loadRawContext(t, dir, manifest.Request.CondContext, contextElements)},
		{"uncond", loadRawContext(t, dir, manifest.Request.UncondContext, contextElements)},
	}
	var worstSeam, worstBlock, worstBranch parityStat
	for _, branch := range branches {
		state, err := trainer.forwardConditioned(sampleIn, branch.context, timeActs.blockE, timeActs.headE)
		if err != nil {
			t.Fatal(err)
		}
		block0 := &state.blocks[0]
		seams := []struct {
			name string
			got  []float32
		}{
			{"self_q_projected", block0.qProj},
			{"self_k_projected", block0.kProj},
			{"self_v_projected", block0.vProj},
			{"self_q_norm", block0.qNorm},
			{"self_k_norm", block0.kNorm},
			{"self_q_rope", block0.qRope},
			{"self_k_rope", block0.kRope},
			{"self_sdpa", block0.attnOut},
			{"self_attn", block0.selfPro},
			{"self_residual", block0.selfRes},
			{"cross_attn", block0.cPro},
			{"cross_residual", block0.crossRes},
			{"ffn", block0.ffnOut},
			{"ffn_residual", block0.output},
		}
		for _, seam := range seams {
			name := fmt.Sprintf("step_00_block_00_%s_%s", branch.name, seam.name)
			stat := requireParity(t, name, seam.got, manifest.trace(t, dir, name), goldenSeamTolerance)
			if stat.Max > worstSeam.Max {
				worstSeam = stat
			}
		}
		for layer := range state.blocks {
			name := fmt.Sprintf("step_00_block_%02d_%s", layer, branch.name)
			stat := requireParity(t, name, state.blocks[layer].output, manifest.trace(t, dir, name), goldenBlockOutputTolerance)
			if stat.Max > worstBlock.Max {
				worstBlock = stat
			}
		}
		name := "step_00_branch_output_" + branch.name
		stat := requireParity(t, name, state.vLatent, manifest.trace(t, dir, name), goldenBranchTolerance)
		if stat.Max > worstBranch.Max {
			worstBranch = stat
		}
	}
	t.Logf("trainer mirror worst: seam=%.6g block=%.6g branch=%.6g", worstSeam.Max, worstBlock.Max, worstBranch.Max)
}
