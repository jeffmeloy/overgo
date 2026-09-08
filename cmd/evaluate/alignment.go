package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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
	if err := manifest.Inspection.Validate(); err != nil {
		return err
	}
	profile, err := speechrecognition.NewAlignmentProfile(manifest.Profile)
	if err != nil {
		return err
	}
	reader, err := dataset.NewAudioPayloadReader(manifest.MemoryBytes)
	if err != nil {
		return err
	}
	defer reader.Close()
	commit, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return err
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	base, err := recipe.RequireDefinition(ctx, store, manifest.BaseRecipe)
	if err != nil {
		return err
	}
	definition, err := modelrecipe.AlignmentDefinition(base, profile.ID)
	if err != nil {
		return err
	}
	commitBatch := func(batch artifact.Batch, err error) error {
		if err != nil {
			return err
		}
		_, err = artifact.CommitBatch(ctx, store, batch)
		if errors.Is(err, artifact.ErrNoChange) {
			return nil
		}
		return err
	}
	if err := commitBatch(profile.Batch("alignment-evaluation/profile/" + profile.ID.String())); err != nil {
		return err
	}
	if err := commitBatch(environment.Batch("alignment-evaluation/environment/" + environment.ID.String())); err != nil {
		return err
	}
	_, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "alignment-evaluation/recipe/"+definition.ID.String(), definition); err != nil {
			return err
		}
	}
	declaration, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/alignment-evaluation-manifest/v1"), manifest)
	if err != nil {
		return err
	}
	if err := commitBatch(artifact.Batch{Key: "alignment-evaluation/manifest/" + declaration.Descriptor.ID.String(), Contents: []artifact.Content{declaration}}, nil); err != nil {
		return err
	}
	session, err := speechrecognition.LoadSession(ctx, store, definition.ID, manifest.MemoryBytes)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))
	lease, err := session.Lease(ctx)
	if err != nil {
		return err
	}
	defer lease.Release()
	type observation struct {
		Reference artifact.ID                `json:"reference"`
		Run       artifact.ID                `json:"run,omitzero"`
		Output    artifact.ID                `json:"output,omitzero"`
		Score     *evaluation.AlignmentScore `json:"score,omitempty"`
		Failure   string                     `json:"failure,omitzero"`
	}
	results := make([]observation, len(manifest.Inputs))
	dependencies := []artifact.ID{declaration.Descriptor.ID, definition.ID, environment.ID}
	failed := 0
	for index, input := range manifest.Inputs {
		result := &results[index]
		result.Reference = input.Reference
		input.Audio.Path = resolveEvaluationPath(filepath.Dir(path), input.Audio.Path)
		attempt := func() error {
			reference, err := speechrecognition.RequireAlignment(ctx, store, input.Reference)
			if err != nil {
				return err
			}
			dependencies = append(dependencies, input.Reference)
			payload, err := reader.Read(ctx, input.Audio, manifest.Inspection.MaximumEncodedBytes)
			if err != nil {
				return err
			}
			predicted, run, err := lease.Align(ctx, payload, input.Audio.Origin, manifest.Inspection, input.Request, speechrecognition.RunBinding{
				Key: fmt.Sprintf("alignment-evaluation/run/%s/%d", declaration.Descriptor.ID, index), CodeCommit: commit, Environment: environment.ID})
			result.Run = run.ID
			if run.ID.Valid() {
				dependencies = append(dependencies, run.ID)
			}
			if err != nil {
				return err
			}
			result.Output = run.Outputs[0]
			score, err := evaluation.ScoreAlignment(reference, predicted)
			if err != nil {
				return err
			}
			result.Score = &score
			return nil
		}
		if err := attempt(); err != nil {
			result.Failure = err.Error()
			failed++
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	report, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/alignment-evaluation-report/v1"), results)
	if err != nil {
		return err
	}
	if err := commitBatch(artifact.Batch{Key: "alignment-evaluation/report/" + report.Descriptor.ID.String(), Contents: []artifact.Content{report}, Lineage: artifact.DependencyLineage(report.Descriptor.ID, dependencies...)}, nil); err != nil {
		return err
	}
	if err := json.NewEncoder(os.Stdout).Encode(struct {
		Report artifact.ID `json:"report"`
		Inputs int         `json:"inputs"`
		Failed int         `json:"failed"`
	}{report.Descriptor.ID, len(results), failed}); err != nil {
		return err
	}
	if failed != 0 {
		return fmt.Errorf("alignment evaluation: %d/%d inputs failed; all attempts retained, no partial pass", failed, len(results))
	}
	return nil
}
