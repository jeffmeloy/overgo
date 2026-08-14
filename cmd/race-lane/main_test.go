package main

import (
	"slices"
	"testing"
)

func TestSanitizerCommandFailsOnHazard(t *testing.T) {
	for _, tool := range deviceRaceTools {
		command := sanitizerCmd(tool, "fixture.test")
		flag := slices.Index(command, "--error-exitcode")
		if flag < 0 || flag+1 >= len(command) || command[flag+1] != "1" {
			t.Errorf("%s command does not fail on a detected hazard: %v", tool, command)
		}
	}
}
