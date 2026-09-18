package main

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const mediaProjectionPath = "docs/media_report.json"

// A rebuildable view, never generation acceptance or another evidence ledger.
type mediaProjection struct {
	Version   uint16                `json:"version"`
	Tasks     []recipe.Task         `json:"tasks"`
	Truncated bool                  `json:"truncated"`
	Counts    mediaProjectionCounts `json:"counts"`
	Rows      []mediaProjectionRow  `json:"rows"`
}

type mediaProjectionCounts struct {
	Activations       int `json:"activations"`
	Stale             int `json:"stale"`
	MeasuredVerifiers int `json:"measured_verifiers"`
}

type mediaProjectionRow struct {
	Name          string                   `json:"name"`
	Model         artifact.ID              `json:"model"`
	Task          recipe.Task              `json:"task"`
	Recipe        artifact.ID              `json:"recipe,omitzero"`
	Activation    artifact.ID              `json:"activation,omitzero"`
	Supersedes    *artifact.ID             `json:"supersedes,omitempty"`
	Tier          string                   `json:"activation_tier"`
	Stale         string                   `json:"stale,omitzero"`
	Verifier      *mediaVerifierProjection `json:"historical_verifier,omitempty"`
	GenerationGap string                   `json:"generation_gap"`
	RepairOwner   string                   `json:"repair_owner"`
}

// Unit-bearing integer fields preserve exact values; nil peak means unmeasured.
// PeakStepDeviceBytes is a maximum of step markers, not a process-peak claim.
type mediaVerifierProjection struct {
	Gate                artifact.ID             `json:"gate"`
	Run                 artifact.ID             `json:"run"`
	Source              string                  `json:"source"`
	Environment         artifact.ID             `json:"environment,omitzero"`
	Inputs              []artifact.ID           `json:"inputs"`
	Outputs             []artifact.ID           `json:"outputs"`
	WallNS              uint64                  `json:"wall_ns"`
	Phases              []runrecord.PhaseMetric `json:"phases"`
	PeakStepDeviceBytes *uint64                 `json:"peak_step_device_bytes,omitempty"`
}

func projectMediaRows(rows []reportRow, scope mediaReportScope, truncated bool) (mediaProjection, error) {
	value := mediaProjection{Version: artifact.InitialDocumentVersion, Tasks: slices.Clone(scope.Tasks), Truncated: truncated, Rows: []mediaProjectionRow{}}
	seen := map[string]bool{}
	for _, row := range rows {
		key := row.modelID.String() + "/" + row.task
		if seen[key] || !slices.Contains(scope.Tasks, recipe.Task(row.task)) {
			return mediaProjection{}, errors.New("media projection: duplicate activation or task outside scope")
		}
		seen[key] = true
		projected := mediaProjectionRow{Name: row.model, Model: row.modelID, Task: recipe.Task(row.task), Recipe: row.recipeID, Tier: row.tier, Stale: row.stale,
			GenerationGap: "Current generation protocol and source compatibility are not bound by activation evidence.", RepairOwner: "modality-verification/media-report"}
		if row.stale != "" {
			if row.verification.Run.ID.Valid() {
				return mediaProjection{}, errors.New("media projection: stale activation carries measurement credit")
			}
			value.Counts.Stale++
		} else {
			projected.Activation, projected.Supersedes = row.activation.Event.ID, row.activation.Event.Supersedes
			var err error
			projected.Verifier, err = projectMediaVerifier(row.verification)
			if err != nil {
				return mediaProjection{}, fmt.Errorf("media projection %s: %w", key, err)
			}
			if projected.Verifier != nil {
				if row.verification.Run.Recipe != row.recipeID || row.activation.Definition.ID != row.recipeID {
					return mediaProjection{}, errors.New("media projection: verifier differs from selected recipe")
				}
				value.Counts.MeasuredVerifiers++
			}
		}
		value.Rows = append(value.Rows, projected)
	}
	value.Counts.Activations = len(value.Rows)
	return value, nil
}

func projectMediaVerifier(verified runrecord.Verification) (*mediaVerifierProjection, error) {
	if !verified.Run.ID.Valid() {
		return nil, nil
	}
	if verified.Run.CodeCommit != verified.Gate.CodeCommit || verified.Run.Recipe != verified.Gate.Recipe || verified.Run.Environment != verified.Gate.Environment {
		return nil, errors.New("verifier source, recipe or environment differs")
	}
	run := verified.Run
	value := &mediaVerifierProjection{Gate: verified.Gate.ID, Run: run.ID, Source: run.CodeCommit, Environment: run.Environment,
		Inputs: slices.Clone(run.Inputs), Outputs: slices.Clone(run.Outputs), WallNS: run.MeasuredNS, Phases: slices.Clone(run.Phases)}
	for _, step := range verified.Gate.Steps {
		var peak *uint64
		for field := range strings.FieldsSeq(step.Evidence) {
			if !strings.HasPrefix(field, "peak_device_bytes=") {
				continue
			}
			bytes, valid := stepPeakDeviceBytes(field)
			if !valid || peak != nil && *peak != bytes {
				return nil, fmt.Errorf("%s: invalid or contradictory peak byte measurement", step.Name)
			}
			peak = &bytes
		}
		if peak != nil && (value.PeakStepDeviceBytes == nil || *peak > *value.PeakStepDeviceBytes) {
			value.PeakStepDeviceBytes = peak
		}
	}
	return value, nil
}

func mediaByteCell(value uint64) string {
	// Binary prefixes: one MiB is 2^20 bytes; one GiB is 2^30 bytes.
	const bytesPerMiB, bytesPerGiB = 1 << 20, 1 << 30
	return fmt.Sprintf("%d B (%.3f MiB; %.3f GiB)", value, float64(value)/bytesPerMiB, float64(value)/bytesPerGiB)
}
