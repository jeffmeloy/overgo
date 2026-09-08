package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

type alignmentManifest struct {
	BaseRecipe  artifact.ID                        `json:"base_recipe"`
	Profile     speechrecognition.AlignmentProfile `json:"profile"`
	MemoryBytes uint64                             `json:"memory_bytes"`
	Inspection  dataset.AudioInspectionPolicy      `json:"inspection"`
	Inputs      []struct {
		Audio     dataset.AudioPayloadReference      `json:"audio"`
		Request   speechrecognition.AlignmentRequest `json:"request"`
		Reference artifact.ID                        `json:"reference"`
	} `json:"inputs"`
}

func evaluateAlignmentManifest(ctx context.Context, repository, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var manifest alignmentManifest
	if err := strictjson.DecodeBytes(data, &manifest); err != nil {
		return err
	}
	if len(manifest.Inputs) == 0 {
		return errors.New("alignment evaluation: empty input denominator")
	}
	profile, err := speechrecognition.NewAlignmentProfile(manifest.Profile)
	if err != nil {
		return err
	}
	execution, err := openSpeechEvaluation(ctx, repository, len(manifest.Inputs), manifest.MemoryBytes, manifest.Inspection)
	if err != nil {
		return err
	}
	defer execution.close()
	base, err := recipe.RequireDefinition(ctx, execution.store, manifest.BaseRecipe)
	if err != nil {
		return err
	}
	definition, err := modelrecipe.AlignmentDefinition(base, profile.ID)
	if err != nil {
		return err
	}
	batch, err := profile.Batch("alignment-evaluation/profile/" + profile.ID.String())
	if err := execution.publish(ctx, batch, err); err != nil {
		return err
	}
	_, published, err := modelrecipe.Status(ctx, execution.store, definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, execution.store, "alignment-evaluation/recipe/"+definition.ID.String(), definition); err != nil {
			return err
		}
	}
	return executeSpeechInputs(ctx, execution, "alignment", manifest, definition.ID, len(manifest.Inputs),
		func(lease *speechrecognition.SpeechLease, index int, binding speechrecognition.RunBinding) (speechObservation[evaluation.AlignmentScore], []artifact.ID, error) {
			input := manifest.Inputs[index]
			result := speechObservation[evaluation.AlignmentScore]{Reference: input.Reference}
			reference, err := speechrecognition.RequireAlignment(ctx, execution.store, input.Reference)
			if err != nil {
				return result, nil, err
			}
			parents := []artifact.ID{input.Reference}
			input.Audio.Path = resolveEvaluationPath(filepath.Dir(path), input.Audio.Path)
			payload, err := execution.reader.Read(ctx, input.Audio, manifest.Inspection.MaximumEncodedBytes)
			if err != nil {
				return result, parents, err
			}
			predicted, run, err := lease.Align(ctx, payload, input.Audio.Origin, manifest.Inspection, input.Request, binding)
			result.Run = run.ID
			if err != nil {
				return result, parents, err
			}
			result.Output = run.Outputs[0]
			score, err := evaluation.ScoreAlignment(reference, predicted)
			if err != nil {
				return result, parents, err
			}
			result.Score = &score
			return result, parents, nil
		})
}
