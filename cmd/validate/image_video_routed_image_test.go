package main

import (
	"overgo/internal/artifact"
)

type mediaRoutedCase struct {
	Case        string      `json:"case"`
	Mode        string      `json:"mode"`
	Run         artifact.ID `json:"run"`
	Output      artifact.ID `json:"output"`
	Acquisition artifact.ID `json:"acquisition"`
	Review      artifact.ID `json:"review"`
	Test        artifact.ID `json:"test_evidence"`
}
