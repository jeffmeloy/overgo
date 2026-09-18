package main

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const imageVideoProtocolPath = "docs/image_video_protocol.json"

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

type mediaProtocolProjection struct {
	Census         artifact.ID                   `json:"inventory_census"`
	RequiredChecks []string                      `json:"required_checks"`
	Cases          []mediaProtocolCaseProjection `json:"cases"`
	Retained       int                           `json:"retained_run_lineages"`
}

type mediaProtocolCaseProjection struct {
	imageVideoCase
	CurrentRecipe artifact.ID `json:"current_recipe,omitzero"`
	Source        string      `json:"source,omitzero"`
	Environment   artifact.ID `json:"environment,omitzero"`
	Retained      bool        `json:"retained_run_lineage"`
	Gap           string      `json:"generation_gap"`
	RepairOwner   string      `json:"repair_owner"`
}

func projectMediaProtocol(ctx context.Context, root string, reader artifact.Reader, rows []mediaProjectionRow) (*mediaProtocolProjection, error) {
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if protocol.Version != artifact.InitialDocumentVersion || protocol.Census.Kind() != artifact.KindEvidence || len(protocol.Cases) == 0 || len(protocol.RequiredChecks) == 0 {
		return nil, errors.New("media protocol: incomplete denominator")
	}
	value := &mediaProtocolProjection{Census: protocol.Census, RequiredChecks: protocol.RequiredChecks, Cases: []mediaProtocolCaseProjection{}}
	seen := map[string]bool{}
	for _, check := range protocol.RequiredChecks {
		if check == "" || seen[check] {
			return nil, errors.New("media protocol: missing or duplicate required check")
		}
		seen[check] = true
	}
	clear(seen)
	for _, declared := range protocol.Cases {
		if declared.ID == "" || seen[declared.ID] || declared.Model.Kind() != artifact.KindModel || declared.Recipe.Kind() != artifact.KindRecipe || declared.Run.Kind() != artifact.KindRun ||
			!slices.Contains([]recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen}, declared.Task) || len(declared.Inputs) == 0 || len(declared.Outputs) == 0 {
			return nil, errors.New("media protocol: invalid or duplicate case")
		}
		seen[declared.ID] = true
		for _, observed := range declared.Observations {
			if !slices.Contains(declared.Outputs, observed.Artifact) || observed.Width <= 0 || observed.Height <= 0 || observed.Frames <= 0 || observed.FPS < 0 || observed.Delay < 0 {
				return nil, errors.New("media protocol: invalid output dimensions or timing units")
			}
		}
		projected := mediaProtocolCaseProjection{imageVideoCase: declared, Gap: "Current source and protocol acceptance are unbound.", RepairOwner: "modality-verification/image-and-video"}
		for _, row := range rows {
			if row.Model == declared.Model && row.Task == declared.Task && row.Stale == "" {
				projected.CurrentRecipe = row.Recipe
			}
		}
		run, err := runrecord.RequireRun(ctx, reader, declared.Run)
		err = cmp.Or(err, checkImageVideoCaseRun(declared, run))
		if err != nil {
			projected.Gap = "Retained run unresolved: " + err.Error()
		} else {
			projected.Retained, projected.Source, projected.Environment = true, run.CodeCommit, run.Environment
			value.Retained++
			if !projected.CurrentRecipe.Valid() {
				projected.Gap = "No healthy current activation for the declared model/task."
			} else if projected.CurrentRecipe != declared.Recipe {
				projected.Gap = "Current activation uses a different recipe; historical scope is retained."
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value.Cases = append(value.Cases, projected)
	}
	return value, nil
}

func writeMediaProtocol(output *bytes.Buffer, value *mediaProtocolProjection) {
	output.WriteString("## Declared generation protocol\n\n")
	if value == nil {
		output.WriteString("No image/video protocol is recorded; generation coverage is unresolved. Repair owner: `modality-verification/image-and-video`.\n\n")
		return
	}
	fmt.Fprintf(output, "[Protocol](%s): %d declared cases; %d retained successful run lineages. These counts establish neither current-source compatibility nor quality acceptance.\n\n", filepath.Base(imageVideoProtocolPath), len(value.Cases), value.Retained)
	output.WriteString("| Case | Original source | Generation gap | Repair owner |\n| --- | --- | --- | --- |\n")
	for _, row := range value.Cases {
		fmt.Fprintf(output, "| `%s` | `%s` | %s | `%s` |\n", escapeMarkdown(row.ID), row.Source, escapeMarkdown(row.Gap), row.RepairOwner)
	}
	output.WriteString("\n")
}
