//go:build windows

package adaptiveparity

import (
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/diffusionimage"
	"overgo/internal/jsonfile"
	"overgo/internal/modeltest"
	"overgo/internal/sensenovarecipe"
	"overgo/internal/testutil"
)

// ownCodePictureScript runs the routed image model's own code and writes the fixture.
const ownCodePictureScript = "scripts/python_golden_references/sensenova_own_picture.py"

// ownCodePictures is what the routed image model's own code drew: the
// text-to-image example of the package shipped beside its weights, for each
// prompt and seed at the size, steps, guidance scale and timestep shift given.
// Type says the arithmetic the model was held in. Means holds each picture as
// the mean of every colour over squares of Square pixels, a byte each, row by
// row.
type ownCodePictures struct {
	Original      string  `json:"original"`
	Type          string  `json:"dtype"`
	Attention     string  `json:"attention"`
	Torch         string  `json:"torch"`
	Width         int     `json:"width"`
	Height        int     `json:"height"`
	Steps         int     `json:"steps"`
	CFGScale      float32 `json:"cfg_scale"`
	TimestepShift float64 `json:"timestep_shift"`
	Square        int     `json:"square"`
	Pictures      []struct {
		Name   string `json:"name"`
		Prompt string `json:"prompt"`
		Seed   int64  `json:"seed"`
		Means  []byte `json:"means"`
	} `json:"pictures"`
}

// ownCodePictureFloor is the agreement, in dB over the fixture's squares, a
// served picture is held to against the model's own. The two served pictures
// agree with it at 37.7 and 39.0 dB. The model's own picture drawn at a
// timestep shift of 1 stands at 20 and 16 dB from its own at 3, without
// guidance at 7 and 9, and from another seed at 5 and 8.
const ownCodePictureFloor = 30

// TestServedInfographicMatchesTheModelsOwnCode holds the picture a request is
// served by SenseNova-U1 to the model's own code: the same prompt, seed, size,
// steps, guidance scale and timestep shift. The runtime under the request is
// held step by step to values the model's own code gave, at 256 by 256 in two
// steps from token ids the reference supplied; this is the path a request
// takes around it, from the prompt's text to the encoded picture, at a size
// whose noise scale is not a power of two.
//
// The size and steps are chosen where the model's own picture is steady, and
// that is not everywhere. Moving one value in a thousand of the model's
// starting noise by one step of its type leaves these two pictures at 39 and
// 42 dB from themselves, so the served ones stand about as far from the
// model's as the model's stand from themselves under the least change a
// rounding makes. At 384 and 512 pixels in eight steps the model's own
// picture in float32 stands 19 to 21 dB from its own in bfloat16, and at its
// recommended 2048 pixels in 50 steps the least change moves it to 24 dB, the
// same poster with other lettering: there a served picture, 21 dB off, cannot
// be told from the model's own by its squares, only by eye.
func TestServedInfographicMatchesTheModelsOwnCode(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "SenseNova-U1-8B-MoT-Infographic-V3")
	var own ownCodePictures
	if err := jsonfile.DecodeStrict(filepath.Join(testutil.RepoRoot(t), "fixtures", "sensenova", "own_picture.json"), &own); err != nil {
		t.Fatalf("the model's own picture is absent (%v); write it with %s", err, ownCodePictureScript)
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	service := pictureServing(t, directory, "own-code-picture")
	for _, picture := range own.Pictures {
		agreement := service.agreement(t, sensenovarecipe.GenerationRequest{
			Prompt: picture.Prompt, Width: own.Width, Height: own.Height, Steps: own.Steps, Seed: picture.Seed,
			CFGScale: own.CFGScale, TimestepShift: own.TimestepShift,
		}, own.Width, own.Height, own.Square, picture.Means)
		t.Logf("%s, seed %d: the served picture against the model's own code's, held in %s, %.2f dB over %d-pixel squares",
			picture.Name, picture.Seed, own.Type, agreement, own.Square)
		if agreement < ownCodePictureFloor {
			t.Errorf("%s: the served picture agrees with the model's own code's at %.2f dB over %d-pixel squares, below %d", picture.Name, agreement, own.Square, ownCodePictureFloor)
		}
	}
}

// upstreamSampleScript runs the diffusion model's upstream sampler and writes the fixture.
const upstreamSampleScript = "scripts/python_golden_references/simplediffusion_upstream_sample.py"

// upstreamSample is what the diffusion model's upstream sampler drew: the
// function sample() of the training script beside its weights, over the
// network of its repository, from the noise overgo draws for Seed, at the size
// and steps given. Means holds the picture as the mean of every colour over
// squares of Square pixels, a byte each, row by row.
type upstreamSample struct {
	Original string `json:"original"`
	Torch    string `json:"torch"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Steps    int    `json:"steps"`
	Seed     int64  `json:"seed"`
	Square   int    `json:"square"`
	Means    []byte `json:"means"`
}

// upstreamSampleFloor is the agreement, in dB over the fixture's squares, a
// served sample is held to against the upstream sampler's. The served sample
// agrees with it at 53 dB. The upstream sampler's own picture in one step
// fewer stands at 42 dB from this one, and from another seed's noise at 23.
const upstreamSampleFloor = 48

// TestServedSampleMatchesTheUpstreamSampler holds the picture a request is
// served by SimpleDiffusion to its upstream sampler: the same size and steps,
// from the same noise. The network's output is held to the upstream network's
// on its real weights at 32 by 32; this is the request's path around it, the
// steps taken along that output and the range the picture is written in, at
// the size and steps of the model's accepted sample.
//
// The noise is overgo's own. The upstream sampler draws torch's, which a seed
// of overgo's does not name, so the reference is given the noise overgo draws
// for the seed and what is held is everything after it. A change to how
// overgo draws its noise needs the fixture written again.
func TestServedSampleMatchesTheUpstreamSampler(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "SimpleDiffusion-TensorProductAttentionRope")
	var upstream upstreamSample
	if err := jsonfile.DecodeStrict(filepath.Join(testutil.RepoRoot(t), "fixtures", "simplediffusion", "upstream_sample.json"), &upstream); err != nil {
		t.Fatalf("the upstream sampler's picture is absent (%v); write it with %s", err, upstreamSampleScript)
	}
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	agreement := pictureServing(t, directory, "upstream-sample").agreement(t,
		diffusionimage.Request{Seed: upstream.Seed, Steps: upstream.Steps, Height: upstream.Height, Width: upstream.Width},
		upstream.Width, upstream.Height, upstream.Square, upstream.Means)
	t.Logf("seed %d: the served sample against the upstream sampler's, %.2f dB over %d-pixel squares", upstream.Seed, agreement, upstream.Square)
	if agreement < upstreamSampleFloor {
		t.Errorf("the served sample agrees with the upstream sampler's at %.2f dB over %d-pixel squares, below %d", agreement, upstream.Square, upstreamSampleFloor)
	}
}
