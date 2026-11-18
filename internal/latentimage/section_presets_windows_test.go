//go:build windows

package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheUltraSectionPresetsHoldTheRecordedSettings holds MOSS's six section presets (moss-ultra-compare/section-presets)
// to the MOSS Ultra workflow's own requests (fixtures/comfy/moss-ultra/requests): each fills a request that sets
// nothing with its section's sampler steps, guidance, sampler and scheduler, denoise, its adapters with their strengths
// as its LoRA loader stacks them, its canvas as its ResolutionSelector sizes it, its sharpening, its skipped blocks (the
// profile's where the section skips layers, none where it has no SkipLayerGuidance node) and the workflow's backup
// negative of an empty exclusion, and a depth section's control the strength and window its MOSSImageControl runs it
// at, a request stating its map alone admitted before its preset fills its control adapter (SQA-409); and a request's
// own settings stand over the preset's, a depth control's end among them.
func TestTheUltraSectionPresetsHoldTheRecordedSettings(t *testing.T) {
	sections := map[string]struct{ slug, sampler string }{
		"ultra-text-to-image": {"text-to-image", "598"}, "ultra-text-to-image-turbo": {"text-to-image-turbo", "331"},
		"ultra-image-to-image": {"image-to-image", "506"}, "ultra-image-to-image-turbo": {"image-to-image-turbo", "198"},
		"ultra-depth-control": {"depth-control", "443"}, "ultra-depth-control-turbo": {"depth-control-turbo", "48"},
	}
	directory := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "model_index.json"), []byte(`{"_class_name": "MOSSImagePipeline"}`), 0o600))
	backup := testcheck.Must(mossEnhancer())(t).NegativeBackup
	for preset, section := range sections {
		var nodes map[string]struct {
			Class  string         `json:"class_type"`
			Inputs map[string]any `json:"inputs"`
		}
		path := filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "requests", section.slug+".json")
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(path))(t), &nodes))
		sampler := nodes[section.sampler].Inputs
		if nodes[section.sampler].Class != "KSampler" {
			t.Fatalf("%s's node %s is %s; want its KSampler", section.slug, section.sampler, nodes[section.sampler].Class)
		}
		var canvas media.Canvas
		var sharpen media.Sharpening
		var adapters []AppliedAdapter
		var recorded *DepthControl
		skips := false
		for _, node := range nodes {
			switch in := node.Inputs; node.Class {
			case "ResolutionSelector":
				aspect, _, _ := strings.Cut(in["aspect_ratio"].(string), " ")
				canvas = media.Canvas{Aspect: aspect, Megapixels: in["megapixels"].(float64), Multiple: int(in["multiple"].(float64))}
			case "ImageSharpen":
				sharpen = media.Sharpening{Radius: int(in["sharpen_radius"].(float64)), Sigma: in["sigma"].(float64), Alpha: in["alpha"].(float64)}
			case "Power Lora Loader (rgthree)":
				for slot := 1; ; slot++ {
					lora, held := in["lora_"+string(rune('0'+slot))].(map[string]any)
					if !held {
						break
					}
					if lora["on"] == true {
						strength := float32(lora["strength"].(float64))
						adapters = append(adapters, AppliedAdapter{Name: strings.TrimSuffix(lora["lora"].(string), ".safetensors"), Strength: &strength})
					}
				}
			case "MOSSImageControl":
				strength, end := float32(in["strength"].(float64)), in["end_percent"].(float64)
				recorded = &DepthControl{Strength: &strength, Start: in["start_percent"].(float64), End: &end}
			case "SkipLayerGuidanceDiTSimple":
				skips = true
			}
		}
		filled := testcheck.Must(Request{Preset: preset}.withPreset(directory))(t)
		size := testcheck.Must(func() ([2]int, error) {
			width, height, err := canvas.Size()
			return [2]int{width, height}, err
		}())(t)
		width, height := size[0], size[1]
		guidance := float32(sampler["cfg"].(float64))
		denoise := sampler["denoise"].(float64)
		if denoise >= 1 {
			denoise = 0
		}
		if filled.Steps != int(sampler["steps"].(float64)) || filled.GuidanceScale == nil || *filled.GuidanceScale != guidance ||
			filled.Schedule == nil || filled.Schedule.Sampler != sampler["sampler_name"] || filled.Schedule.Name != sampler["scheduler"] ||
			filled.Denoise != denoise || filled.Width != width || filled.Height != height || filled.Sharpen == nil || *filled.Sharpen != sharpen ||
			filled.NegativePrompt != backup {
			t.Errorf("%s fills %+v; want %s's KSampler %v, its canvas %+v (%dx%d), its sharpening %+v and the backup negative", preset, filled, section.slug, sampler, canvas, width, height, sharpen)
		}
		applied := filled.applied()
		if !slices.EqualFunc(applied, adapters, func(a, b AppliedAdapter) bool { return a.Name == b.Name && a.strength() == b.strength() }) {
			t.Errorf("%s applies %+v; want %s's loader's %+v", preset, applied, section.slug, adapters)
		}
		if skips != (filled.SkipBlocks == nil) || !skips && len(*filled.SkipBlocks) != 0 {
			t.Errorf("%s skips %v; want the profile's blocks where %s skips layers (%t), none where it does not", preset, filled.SkipBlocks, section.slug, skips)
		}
		if recorded == nil {
			continue
		}
		// A depth section's request states its map alone: the director admits it before its preset fills its control
		// adapter, and the preset fills the control's window as the section's MOSSImageControl runs it.
		depth := Request{Prompt: "a fox", Width: 1024, Height: 576, Preset: preset, Control: &DepthControl{Map: "a depth map"}}
		testcheck.NoError(t, ValidateRequest(depth))
		drawn := testcheck.Must(depth.withPreset(directory))(t).Control
		if drawn.strength() != *recorded.Strength || drawn.Start != recorded.Start || drawn.end() != *recorded.End {
			t.Errorf("%s runs its depth control at %g from %g to %g; want %s's MOSSImageControl, %g from %g to %g",
				preset, drawn.strength(), drawn.Start, drawn.end(), section.slug, *recorded.Strength, recorded.Start, *recorded.End)
		}
	}
	whole := 1.0
	stated := testcheck.Must(Request{Preset: "ultra-depth-control", Control: &DepthControl{Map: "a depth map", End: &whole}}.withPreset(directory))(t)
	if stated.Control.end() != whole {
		t.Errorf("a depth control stating its end under the depth preset ends at %g; want its own %g", stated.Control.end(), whole)
	}
	if ValidateRequest(Request{Prompt: "a fox", Width: 1024, Height: 576, Control: &DepthControl{Map: "a depth map"}}) == nil {
		t.Error("a depth request naming neither a preset nor an adapter was admitted; want it refused for its control adapter")
	}
	own := float32(2)
	mine := testcheck.Must(Request{Preset: "ultra-text-to-image", Steps: 20, GuidanceScale: &own, Width: 1024, Height: 576, NegativePrompt: "blur"}.withPreset(directory))(t)
	if mine.Steps != 20 || *mine.GuidanceScale != own || mine.Width != 1024 || mine.Height != 576 || mine.NegativePrompt != "blur" || mine.Schedule == nil {
		t.Errorf("a request's own settings under the preset came back as %+v; want its own steps, guidance, size and negative, the preset's schedule", mine)
	}
}
