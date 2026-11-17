//go:build windows

package mediacapability

import (
	"slices"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// TestEveryMediaPageOffersTheCommonControlsInOneOrder holds every generation page to one set of controls
// (common-controls/do, the owner's direction of 2026-11-16): each request a page describes lists the fields that name
// a common control first, in the roles' order, behind the page's disclosure exactly where the role is tuning, and
// every field that names none after them, behind it. Guidance strength, guidance_scale, cfg_scale or guide_scale by
// model, is "guidance" on each page, and a request's prompt, negative prompt, size, frames and seed are one role
// wherever a request takes them.
func TestEveryMediaPageOffersTheCommonControlsInOneOrder(t *testing.T) {
	t.Parallel()
	pages := []recipe.ModuleID{
		modelrecipe.ModuleOscillatorImagePrepare, modelrecipe.ModuleOscillatorVideoPrepare, modelrecipe.ModuleDiffusionImagePrepare,
		modelrecipe.ModuleRAEImagePrepare, modelrecipe.ModuleLatentVideoPrepare, modelrecipe.ModuleReferenceVideoPrepare,
		modelrecipe.ModuleLatentImagePrepare, modelrecipe.ModuleRoutedImagePrepare,
	}
	common := map[string]string{"prompt": "prompt", "negative_prompt": "negative prompt", "width": "width", "height": "height", "frames": "frames", "seed": "seed"}
	for _, page := range pages {
		controls, refusal := Controls(page, "")
		if refusal != "" {
			t.Errorf("%s: %s", page, refusal)
			continue
		}
		last := 0
		for _, control := range controls {
			rank := slices.Index(controlRoles, control.Role)
			if rank < 0 && control.Role != "" {
				t.Errorf("%s's %s names the role %q, no common control", page, control.Name, control.Role)
			}
			if rank < 0 {
				rank = len(controlRoles)
			}
			if rank < last {
				t.Errorf("%s lists %s (%q) after a later role", page, control.Name, control.Role)
			}
			last = rank
			if control.Advanced != (rank >= tuningRoles) {
				t.Errorf("%s's %s (%q) is behind the disclosure %t; a role is tuning from steps on, and a field with no role is", page, control.Name, control.Role, control.Advanced)
			}
			if role, named := common[control.Name]; named && control.Role != role {
				t.Errorf("%s's %s names the role %q; it is %q on every page", page, control.Name, control.Role, role)
			}
		}
	}
	for page, field := range map[recipe.ModuleID]string{
		modelrecipe.ModuleLatentImagePrepare: "guidance_scale", modelrecipe.ModuleRoutedImagePrepare: "cfg_scale", modelrecipe.ModuleLatentVideoPrepare: "guide_scale",
	} {
		controls, _ := Controls(page, "")
		if at := slices.IndexFunc(controls, func(c Control) bool { return c.Name == field }); at < 0 || controls[at].Role != "guidance" {
			t.Errorf("%s's guidance strength %s is not the guidance control: %+v", page, field, controls)
		}
	}
}
