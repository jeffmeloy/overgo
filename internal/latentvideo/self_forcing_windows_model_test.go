//go:build windows && modeltest

package latentvideo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// The student against its own code, from the first measured run of 2026-11-08 less a margin. Given its own code's
// text rows and noises, its first chunk ran at cosine 0.9909 and its 21 latent frames at 0.878 with normalised RMS
// 0.495; overgo's text context ran at 0.99978 against its own code's. The chunks after the first read the ones before
// them and drift: overgo against itself, its text context moved by that 0.0002, ran at 0.9988 in its first chunk and
// 0.914 in its last, so the whole clip's floor is the looser one.
const (
	selfForcingFirstChunkCosine = 0.985
	selfForcingGoldenCosine     = 0.85
	selfForcingGoldenNRMS       = 0.55
	selfForcingTextCosine       = 0.999
)

// TestSelfForcingMatchesItsUpstreamGolden holds the Self-Forcing student, run through LiveEdit's chunked sampler with
// no source channels, to its own code's generation: scripts/python_golden_references/self_forcing_own_generation.py
// ran Self-Forcing's inference.py steps on one prompt at seed 0 and recorded the initial noise, each noise drawn
// between steps, the text encoder's rows and the final latents of 21 latent frames, 81 frames at 832 x 480. The
// runtime's seeded noise is torch's on the device, so the same seed draws the same noises; given the recorded text rows
// through the student's own projection, its first chunk and its whole clip run along the recorded latents above their
// floors, overgo's own text context runs along the recorded one, and the 81 frames decode at the size the student was
// trained at.
func TestSelfForcingMatchesItsUpstreamGolden(t *testing.T) {
	t.Parallel()
	cudatest.Require(t)
	golden := filepath.Join(testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t).References(), "self-forcing-golden")
	var record struct {
		Prompt       string `json:"prompt"`
		Seed         int64  `json:"seed"`
		LatentFrames int    `json:"latent_frames"`
		TextTokens   int    `json:"text_tokens"`
		TextWidth    int    `json:"text_width"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(golden, "record.json")))(t), &record))
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	checkpoint := filepath.Join(modeltest.Directory(t, "Self-Forcing"), "checkpoints", "self_forcing_dmd.pt")
	policy := testcheck.Must(checkpointEditPolicy(checkpoint))(t)
	profile := testcheck.Must(ResolveProfile(wan))(t)
	condition := testcheck.Must(chunkCausalCondition(t.Context(), wan, checkpoint, policy, profile, record.Prompt))(t)
	base := testcheck.Must(LoadDenoiserConfig(wan, profile.Policy))(t)
	// The upstream text rows through the student's own projection: the context its own code's generator read.
	student := testcheck.Must(CompileReferenceEditCheckpoint(checkpoint, base))(t)
	projection := testcheck.Must(func() (projectionWeights, error) {
		weights, _, err := student.loadTextProjection()
		return weights, err
	}())(t)
	rows := readRetainedDenoiserF32(t, filepath.Join(golden, "text.f32"), record.TextTokens*record.TextWidth)
	upstream := testcheck.Must(projectCompactTextConditioning(rows, record.TextTokens, base.TextLen, record.TextWidth, base.Dim, projection, true))(t)
	textCosine, textNRMS := latentAgreement(condition.TextContext, upstream)
	t.Logf("overgo's text context: cosine %.6f, normalised RMS %.4f against its own code's", textCosine, textNRMS)
	if textCosine < selfForcingTextCosine {
		t.Fatalf("overgo's text context runs at cosine %.6f against its own code's, below %.4f", textCosine, selfForcingTextCosine)
	}
	frames := (record.LatentFrames-1)*profile.Policy.VAEStride[0] + 1
	runtime := testcheck.Must(NewReferenceEditRuntime(ReferenceEditRuntimeConfig{
		WanDirectory: wan, EditCheckpoint: checkpoint, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Source:         SourceVideoShape{Channels: 3, Frames: frames, Height: policy.SourceHeight, Width: policy.SourceWidth},
		FramesPerChunk: condition.FramesPerChunk, LocalAttentionFrames: condition.LocalAttention,
		Timesteps: condition.Timesteps, Sigmas: condition.Sigmas, ContextTimestep: condition.ContextTimestep, Warp: condition.Warp,
		Layers: base.NumLayers, DeviceOrdinal: 0, CodecLibrary: true,
	}))(t)
	t.Cleanup(func() { testcheck.NoError(t, runtime.Close()) })
	decoded := 0
	result := testcheck.Must(runtime.Run(t.Context(), upstream, record.Seed, nil, nil, func(_ int, _ []float32, height, width int) error {
		if height != policy.SourceHeight || width != policy.SourceWidth {
			t.Fatalf("a frame decoded at %d x %d, want %d x %d", width, height, policy.SourceWidth, policy.SourceHeight)
		}
		decoded++
		return nil
	}))(t)
	if decoded != frames {
		t.Fatalf("the student decoded %d frames, want %d", decoded, frames)
	}
	latent := runtime.editPlan.Latent
	spatial := latent.LatentHeight * latent.LatentWidth
	frameMajor := readRetainedDenoiserF32(t, filepath.Join(golden, "latents.f32"), len(result.Latent))
	want := make([]float32, len(frameMajor))
	for frame := range latent.LatentFrames {
		for channel := range latent.Channels {
			from := (frame*latent.Channels + channel) * spatial
			to := (channel*latent.LatentFrames + frame) * spatial
			copy(want[to:to+spatial], frameMajor[from:from+spatial])
		}
	}
	cosine, nrms := latentAgreement(result.Latent, want)
	for chunk := range latent.LatentFrames / condition.FramesPerChunk {
		var got, held []float32
		for channel := range latent.Channels {
			from := (channel*latent.LatentFrames + chunk*condition.FramesPerChunk) * spatial
			got = append(got, result.Latent[from:from+condition.FramesPerChunk*spatial]...)
			held = append(held, want[from:from+condition.FramesPerChunk*spatial]...)
		}
		chunkCosine, chunkNRMS := latentAgreement(got, held)
		t.Logf("chunk %d: cosine %.6f, normalised RMS %.4f", chunk, chunkCosine, chunkNRMS)
		if chunk == 0 && chunkCosine < selfForcingFirstChunkCosine {
			t.Fatalf("the student's first chunk runs at cosine %.6f against its own code's, below %.4f", chunkCosine, selfForcingFirstChunkCosine)
		}
	}
	t.Logf("the student's latents: cosine %.6f, normalised RMS %.4f against its own code's; denoised in %.1f s, decoded in %.1f s",
		cosine, nrms, result.DenoiseWallSec, result.Decode.DecodeWallSec)
	if cosine < selfForcingGoldenCosine || nrms > selfForcingGoldenNRMS {
		t.Fatalf("the student's latents run at cosine %.6f, normalised RMS %.4f against its own code's, below the floor of %.4f and %.4f",
			cosine, nrms, selfForcingGoldenCosine, selfForcingGoldenNRMS)
	}
}
