package main

import (
	"fmt"
	"slices"
)

type nativeTextCase struct {
	Name      string  `json:"name"`
	Prompt    string  `json:"prompt"`
	MaxTokens int     `json:"max_tokens"`
	Chat      bool    `json:"chat"`
	Input     []int32 `json:"input"`
	Output    []int32 `json:"output"`
	Text      string  `json:"text"`
}

type nativeTextCapture struct {
	Cases []nativeTextCase `json:"cases"`
}

func checkNativeTextCaptures(goCapture, native nativeTextCapture, expectedInputs, expectedOutputs int) error {
	names := []string{"capital", "count", "code", "integer-addition"}
	if len(goCapture.Cases) != len(names) || len(native.Cases) != len(names) {
		return fmt.Errorf("native text reference: incomplete case denominator")
	}
	inputs, outputs := 0, 0
	for i, name := range names {
		a, b := goCapture.Cases[i], native.Cases[i]
		if a.Name != name || b.Name != name || a.Prompt == "" || len(a.Input) == 0 || len(a.Output) == 0 || len(a.Output) > a.MaxTokens ||
			!slices.Equal(a.Input, b.Input) || !slices.Equal(a.Output, b.Output) || a.Text != b.Text {
			return fmt.Errorf("native text reference: case %s differs", name)
		}
		inputs += len(a.Input)
		outputs += len(a.Output)
	}
	if inputs != expectedInputs || outputs != expectedOutputs {
		return fmt.Errorf("native text reference: token denominator differs")
	}
	return nil
}
