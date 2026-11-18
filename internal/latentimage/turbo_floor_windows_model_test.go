//go:build modeltest && windows

package latentimage

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// turboFloors are the Turbo sections' floors (moss-eight-step), each its first measured run's agreement with ComfyUI's
// (2026-11-18) less goldenFloorMargin, as goldenFloors are set; their changed inputs kept -1.23 and -2.06 dB. Text to
// image Turbo is held once recorded on a real prompt (moss-real-prompt): on the workflow's empty prompt its latent kept
// -1.54 dB and a reseeded draw -2.19, too near for a floor 3 dB down to hold the input's part.
var turboFloors = map[string]goldenFloor{
	"image to image Turbo": {latent: 7.17 - goldenFloorMargin, picture: 20.70 - goldenFloorMargin},
	"depth control Turbo":  {latent: 1.93 - goldenFloorMargin, picture: 13.44 - goldenFloorMargin},
}

// turboSection is a Turbo section's request as its recording asks it at 1024 x 576: its prompts, sampler, sharpening
// (textSection) and the adapters its LoRA loader stacks, each switched on at its strength, in the loader's order.
func turboSection(t *testing.T, recording comfySection, sampler, positive, negative, sharpen, loader string) Request {
	t.Helper()
	request := textSection(t, recording, sampler, positive, negative, sharpen)
	slots := recording.inputs(loader)
	var adapters []AppliedAdapter
	for slot := 1; ; slot++ {
		lora, held := slots[fmt.Sprintf("lora_%d", slot)].(map[string]any)
		if !held {
			break
		}
		if lora["on"] == true {
			strength := float32(lora["strength"].(float64))
			adapters = append(adapters, AppliedAdapter{Name: strings.TrimSuffix(lora["lora"].(string), ".safetensors"), Strength: &strength})
		}
	}
	if len(adapters) == 0 {
		t.Fatalf("the recording's loader %s switches on no adapter: %v", loader, slots)
	}
	request.Adapters = &adapters
	return request
}

// mirroredPicture is a request whose picture, or depth control's picture, is mirrored: the control of a section drawn
// from a picture.
func mirroredPicture(t *testing.T, request Request) Request {
	t.Helper()
	if request.Image != "" {
		request.Image = base64.StdEncoding.EncodeToString(mirrored(t, testcheck.Must(base64.StdEncoding.DecodeString(request.Image))(t)))
	}
	if request.Control != nil {
		control := *request.Control
		control.Picture = base64.StdEncoding.EncodeToString(mirrored(t, testcheck.Must(base64.StdEncoding.DecodeString(control.Picture))(t)))
		request.Control = &control
	}
	return request
}

// turboCases are the MOSS Ultra workflow's Turbo sections drawn from a picture, each 8 steps at guidance 1 through the
// adapters its LoRA loader stacks (DMD2 and the SFT adapter, moss-control before them for depth), recorded at 1024 x
// 576: image to image at denoise 0.72 from the recorder's picture, and depth control of the recorder's picture, its map
// made as the section makes it (sectionDepth) over the first half of the sampling, skipping no block. Text to image
// Turbo (its sampler 331, prompts 27 and 325, sharpening 336, loader 341, picture 147) waits for a real prompt's
// recording (turboFloors).
var turboCases = []goldenCase{
	{name: "image to image Turbo", run: "pass-1", slug: "image-to-image-turbo", sampler: "198", finished: "344", control: mirroredPicture,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["8.image"] != "overgo-recorder-input.png" || recording.Set["685.megapixels"] != 0.5625 {
				t.Fatalf("the recording set %v; want the recorder's picture at 1024 x 576", recording.Set)
			}
			request := turboSection(t, recording, "198", "316", "317", "343", "314")
			request.Image, request.Denoise = base64.StdEncoding.EncodeToString(ultraPictureBytes(t)), recording.inputs("198")["denoise"].(float64)
			return request
		}},
	{name: "depth control Turbo", run: "pass-1", slug: "depth-control-turbo", sampler: "48", finished: "364", control: mirroredPicture,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			estimator, control := recording.inputs("366:41"), recording.inputs("366:46")
			if recording.Set["215.image"] != "overgo-recorder-input.png" || recording.Set["695.megapixels"] != 0.5625 || estimator["resolution"] != float64(depthResolution) {
				t.Fatalf("the recording set %v with an estimator of %v; want the recorder's picture at 1024 x 576, its depth at %d", recording.Set, estimator, depthResolution)
			}
			request := turboSection(t, recording, "48", "361", "362", "363", "422")
			strength, end, none := float32(control["strength"].(float64)), control["end_percent"].(float64), []int{}
			request.SkipBlocks = &none
			request.Control = &DepthControl{Picture: base64.StdEncoding.EncodeToString(ultraPictureBytes(t)), Strength: &strength, Start: control["start_percent"].(float64), End: &end}
			return request
		}},
}

// TestTheTurboSectionsHoldTheirGoldenFloors holds overgo's MOSS on the MOSS Ultra workflow's own files to the
// workflow's recordings of its Turbo sections drawn from a picture (moss-eight-step), as
// TestUltraSectionsHoldTheirGoldenFloors holds its sections of 50 steps: each, asked as its recording asks it, keeps its
// final latent and finished picture above their floors of agreement with ComfyUI's, and with its picture mirrored keeps
// its latent below.
func TestTheTurboSectionsHoldTheirGoldenFloors(t *testing.T) {
	cudatest.Require(t)
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	profile := testcheck.Must(ResolveProfile(root))(t)
	holdGoldenFloors(t, root, profile, turboCases, turboFloors)
}
