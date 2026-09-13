package main

import (
	"errors"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"slices"
)

type imageVideoProtocol struct {
	Version         uint16           `json:"version"`
	Census          artifact.ID      `json:"inventory_census"`
	Cases           []imageVideoCase `json:"cases"`
	RequiredChecks  []string         `json:"required_checks"`
	ReviewQuestions []string         `json:"review_questions"`
	Rules           []string         `json:"rules"`
}

type imageVideoCase struct {
	ID           string                  `json:"id"`
	Model        artifact.ID             `json:"model"`
	Task         recipe.Task             `json:"task"`
	Recipe       artifact.ID             `json:"recipe"`
	Run          artifact.ID             `json:"source_run"`
	Inputs       []artifact.ID           `json:"inputs"`
	Outputs      []artifact.ID           `json:"outputs"`
	Scope        string                  `json:"scope"`
	Observations []imageVideoObservation `json:"observations"`
}

type imageVideoObservation struct {
	Artifact  artifact.ID `json:"artifact"`
	Width     int         `json:"width"`
	Height    int         `json:"height"`
	Frames    int         `json:"frames"`
	FPS       int         `json:"fps,omitzero"`
	Delay     int         `json:"delay_centiseconds,omitzero"`
	FPSSource string      `json:"fps_source,omitzero"`
}

func checkImageVideoCaseRun(value imageVideoCase, run runrecord.Run) error {
	if run.ID != value.Run || run.Recipe != value.Recipe || run.Outcome != runrecord.OutcomeSucceeded ||
		len(value.Inputs) == 0 || len(value.Outputs) == 0 ||
		!slices.Equal(value.Inputs, run.Inputs) || !slices.Equal(value.Outputs, run.Outputs) {
		return errors.New("media protocol: retained run lineage differs")
	}
	return nil
}
