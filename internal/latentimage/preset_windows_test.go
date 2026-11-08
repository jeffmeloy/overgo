package latentimage

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestARequestRunsTheNamedPresetOfItsProfile holds a picture request to the recipe it names: on 2026-11-08 each
// profile held one recipe, the authors', and another worth serving (MOSS's 25 steps at guidance 2.5 from ComfyUI's
// tutorial, Krea's 4-step draft) was a code change. A request naming a preset takes its steps and guidance where it
// states none, its own stated value standing, and comes out as the request that states them; one naming none is the
// authors' recipe as it was; one naming a preset its model does not declare is refused.
func TestARequestRunsTheNamedPresetOfItsProfile(t *testing.T) {
	t.Parallel()
	directory := func(pipeline string) string {
		root := t.TempDir()
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, "model_index.json"), []byte(`{"_class_name":"`+pipeline+`"}`), 0o644))
		return root
	}
	moss, krea := directory("MOSSImagePipeline"), directory("Krea2Pipeline")
	tutorial := testcheck.Must(Request{Prompt: "a fox", Preset: "comfy-tutorial"}.withPreset(moss))(t)
	if tutorial.Steps != 25 || tutorial.GuidanceScale == nil || *tutorial.GuidanceScale != 2.5 || tutorial.Preset != "" {
		t.Fatalf("MOSS's tutorial preset gave %+v, want 25 steps at guidance 2.5 and no name left", tutorial)
	}
	stated := float32(4)
	own := testcheck.Must(Request{Prompt: "a fox", Preset: "comfy-tutorial", Steps: 30, GuidanceScale: &stated}.withPreset(moss))(t)
	if own.Steps != 30 || *own.GuidanceScale != stated {
		t.Fatalf("a request's own steps and guidance gave way to its preset's: %+v", own)
	}
	if draft := testcheck.Must(Request{Prompt: "a fox", Preset: "draft"}.withPreset(krea))(t); draft.Steps != 4 || draft.GuidanceScale != nil {
		t.Fatalf("Krea's draft gave %+v, want 4 steps and the distilled model's own guidance", draft)
	}
	reference := Request{Prompt: "a fox", Steps: 8}
	if named := testcheck.Must(reference.withPreset(krea))(t); named != reference {
		t.Fatalf("a request naming no preset became %+v", named)
	}
	if _, err := (Request{Prompt: "a fox", Preset: "draft"}).withPreset(moss); err == nil {
		t.Fatal("MOSS ran a preset only Krea declares")
	}
}
