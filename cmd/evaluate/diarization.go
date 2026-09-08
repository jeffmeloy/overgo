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
	"overgo/internal/recipecontract"
	"overgo/internal/speechactivity"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

type diarizationManifest struct {
	Recipe      artifact.ID                       `json:"recipe,omitzero"`
	Profile     *speechrecognition.SpeakerProfile `json:"profile,omitempty"`
	MemoryBytes uint64                            `json:"memory_bytes"`
	Inspection  dataset.AudioInspectionPolicy     `json:"inspection"`
	Inputs      []struct {
		Audio     dataset.AudioPayloadReference `json:"audio"`
		Reference artifact.ID                   `json:"reference"`
		Span      recipecontract.SampleSpan     `json:"span"`
		Alignment artifact.ID                   `json:"alignment,omitzero"`
		Activity  artifact.ID                   `json:"activity,omitzero"`
	} `json:"inputs"`
}

func evaluateDiarizationManifest(ctx context.Context, repository, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var manifest diarizationManifest
	if err := strictjson.DecodeBytes(data, &manifest); err != nil {
		return err
	}
	var derived recipe.Definition
	if manifest.Profile != nil {
		profile, err := speechrecognition.NewSpeakerProfile(*manifest.Profile)
		if err != nil {
			return err
		}
		derived, err = modelrecipe.DiarizationDefinition(profile.Model, profile.ID, profile.Inventory)
		if err != nil {
			return err
		}
		if manifest.Recipe.Valid() && manifest.Recipe != derived.ID {
			return errors.New("diarization evaluation: recipe pin differs from profile")
		}
		manifest.Profile = &profile
		manifest.Recipe = derived.ID
	}
	if manifest.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("diarization evaluation: recipe or bound profile required")
	}
	execution, err := openSpeechEvaluation(ctx, repository, len(manifest.Inputs), manifest.MemoryBytes, manifest.Inspection)
	if err != nil {
		return err
	}
	defer execution.close()
	if manifest.Profile != nil {
		batch, err := manifest.Profile.Batch("diarization-evaluation/profile/" + manifest.Profile.ID.String())
		if err := execution.publish(ctx, batch, err); err != nil {
			return err
		}
		_, published, err := modelrecipe.Status(ctx, execution.store, derived.ID)
		if err != nil {
			return err
		}
		if !published {
			if _, _, err := modelrecipe.PublishCandidate(ctx, execution.store, "diarization-evaluation/recipe/"+derived.ID.String(), derived); err != nil {
				return err
			}
		}
	}
	definition, err := recipe.RequireDefinition(ctx, execution.store, manifest.Recipe)
	if err != nil {
		return err
	}
	if definition.Task != recipe.TaskDiarization {
		return errors.New("diarization evaluation: task differs")
	}
	return executeSpeechInputs(ctx, execution, "diarization", manifest, definition.ID, len(manifest.Inputs),
		func(lease *speechrecognition.SpeechLease, index int, binding speechrecognition.RunBinding) (speechObservation[evaluation.SpeechTurnScore], []artifact.ID, error) {
			input := manifest.Inputs[index]
			result := speechObservation[evaluation.SpeechTurnScore]{Reference: input.Reference}
			reference, err := speechrecognition.RequireSpeechTurns(ctx, execution.store, input.Reference)
			if err != nil {
				return result, nil, err
			}
			parents := []artifact.ID{input.Reference}
			input.Audio.Path = resolveEvaluationPath(filepath.Dir(path), input.Audio.Path)
			payload, err := execution.reader.Read(ctx, input.Audio, manifest.Inspection.MaximumEncodedBytes)
			if err != nil {
				return result, parents, err
			}
			predicted, run, err := lease.Diarize(ctx, payload, input.Audio.Origin, manifest.Inspection, binding)
			result.Run = run.ID
			if err != nil {
				return result, parents, err
			}
			result.Output = run.Outputs[0]
			score, err := evaluation.ScoreSpeechTurns(ctx, reference, predicted, input.Span, manifest.MemoryBytes)
			if err != nil {
				return result, parents, err
			}
			result.Score = &score
			if input.Alignment.Valid() {
				alignment, err := speechrecognition.RequireAlignment(ctx, execution.store, input.Alignment)
				if err != nil {
					return result, parents, err
				}
				parents = append(parents, input.Alignment)
				var activity *recipecontract.ActivitySegments
				if input.Activity.Valid() {
					segments, err := speechactivity.RequireSegments(ctx, execution.store, input.Activity)
					if err != nil {
						return result, parents, err
					}
					activity = &segments
					parents = append(parents, input.Activity)
				}
				words, err := speechrecognition.AttributeWords(ctx, alignment, predicted, activity)
				if err != nil {
					return result, parents, err
				}
				content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/speaker-attributed-words/v1"), struct {
					Alignment artifact.ID                        `json:"alignment"`
					Turns     artifact.ID                        `json:"turns"`
					Activity  artifact.ID                        `json:"activity,omitzero"`
					Words     []speechrecognition.AttributedWord `json:"words"`
				}{input.Alignment, result.Output, input.Activity, words})
				if err != nil {
					return result, parents, err
				}
				dependencies := []artifact.ID{input.Alignment, result.Output}
				if input.Activity.Valid() {
					dependencies = append(dependencies, input.Activity)
				}
				if err := execution.publish(ctx, artifact.Batch{Key: "diarization-evaluation/words/" + content.Descriptor.ID.String(), Contents: []artifact.Content{content}, Lineage: artifact.DependencyLineage(content.Descriptor.ID, dependencies...)}, nil); err != nil {
					return result, parents, err
				}
				result.Attribution = content.Descriptor.ID
			} else if input.Activity.Valid() {
				return result, parents, errors.New("diarization evaluation: activity composition requires alignment")
			}
			return result, parents, nil
		})
}
