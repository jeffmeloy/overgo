package main

import (
	"testing"

	"overgo/internal/recipe"
)

func TestCommandsRegisterExecutableCapabilities(t *testing.T) {
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen,
	} {
		t.Run(string(task), func(t *testing.T) {
			capability, ok := capabilityCommands[task]
			if !ok || capability.inventory == nil || capability.definition == nil || capability.execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
		})
	}
}
