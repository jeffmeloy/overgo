package main

import (
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/recipe"
)

func TestCommandsRegisterExecutableCapabilities(t *testing.T) {
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen,
	} {
		t.Run(string(task), func(t *testing.T) {
			capability, ok := capabilityruntime.Lookup(task)
			if !ok || capability.Inventory == nil || capability.Execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
		})
	}
}
