//go:build modeltest

package main

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestEveryGeneratorHasAUserSizeCell holds every served image and video generator to a timing at the size a person
// asks for: on 2026-11-08 Wan alone had one, and Krea-2-Turbo, MOSS-Image-SFT and SenseNova were timed at a smoke
// request's size, where the network does almost no work. Each generator's cells hold a user-size cell, and its small
// verified request is measured at its publisher's size, which its page offers too: 2048 by 2048 for Krea-2-Turbo and
// SenseNova, 1024 by 1024 for MOSS-Image-SFT, 81 frames at 832 by 480 for Wan.
func TestEveryGeneratorHasAUserSizeCell(t *testing.T) {
	t.Parallel()
	for _, generator := range []struct {
		name          string
		task          recipe.Task
		entry         recipe.ModuleID
		width, height int
		cell          string
	}{
		{"Krea-2-Turbo", recipe.TaskImageGen, modelrecipe.ModuleLatentImagePrepare, 2048, 2048, imageUserSize},
		{"MOSS-Image-SFT-797fa8e678b4f", recipe.TaskImageGen, modelrecipe.ModuleLatentImagePrepare, 1024, 1024, imageUserSize},
		{"SenseNova-U1-8B-MoT-Infographic-V3", recipe.TaskImageGen, modelrecipe.ModuleRoutedImagePrepare, 2048, 2048, imageUserSize},
		{"Wan2.1-T2V-1.3B", recipe.TaskVideoGen, modelrecipe.ModuleLatentVideoPrepare, 832, 480, videoUserSize},
	} {
		t.Run(generator.name, func(t *testing.T) {
			t.Parallel()
			if cells := kindValidation(generator.task); !slices.Contains(cells, generator.cell) {
				t.Fatalf("%s's cells are %v, with no %s", generator.name, cells, generator.cell)
			}
			directory := modeltest.Directory(t, generator.name)
			sized, lifted, err := mediacapability.PersonRequest(generator.entry, directory, `{"prompt":"a fox","width":32,"height":32,"frames":5,"steps":1}`)
			var fields struct{ Width, Height int }
			testcheck.NoError(t, json.Unmarshal([]byte(sized), &fields))
			if err != nil || !lifted || fields.Width != generator.width || fields.Height != generator.height {
				t.Fatalf("%s is measured at %s (%v), want %d by %d", generator.name, sized, err, generator.width, generator.height)
			}
		})
	}
}
