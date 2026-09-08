package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

// speechEvaluation owns the common source identity, read workspace and store
// lifetime for native alignment and diarization evaluations.
type speechEvaluation struct {
	store       *overgodb.Store
	reader      *dataset.AudioPayloadReader
	commit      string
	environment runrecord.Environment
	memory      uint64
}

func openSpeechEvaluation(ctx context.Context, repository string, count int, memory uint64, inspection dataset.AudioInspectionPolicy) (*speechEvaluation, error) {
	if count == 0 {
		return nil, errors.New("speech evaluation: empty input denominator")
	}
	if err := inspection.Validate(); err != nil {
		return nil, err
	}
	reader, err := dataset.NewAudioPayloadReader(memory)
	if err != nil {
		return nil, err
	}
	commit, err := modelintake.CleanRevision(ctx)
	if err != nil {
		reader.Close()
		return nil, err
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		reader.Close()
		return nil, err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		reader.Close()
		return nil, err
	}
	e := &speechEvaluation{store: store, reader: reader, commit: commit, environment: environment, memory: memory}
	batch, err := environment.Batch("speech-evaluation/environment/" + environment.ID.String())
	if err := e.publish(ctx, batch, err); err != nil {
		e.close()
		return nil, err
	}
	return e, nil
}

func (e *speechEvaluation) close() error { return errors.Join(e.reader.Close(), e.store.Close()) }

func (e *speechEvaluation) publish(ctx context.Context, batch artifact.Batch, err error) error {
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, e.store, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}

type speechObservation[S any] struct {
	Reference   artifact.ID `json:"reference"`
	Run         artifact.ID `json:"run,omitzero"`
	Output      artifact.ID `json:"output,omitzero"`
	Attribution artifact.ID `json:"attribution,omitzero"`
	Score       *S          `json:"score,omitempty"`
	Failure     string      `json:"failure,omitzero"`
}

// executeSpeechInputs records every attempted input, including failures before
// a model run exists. Neither failed cases nor zero inputs can become a pass.
func executeSpeechInputs[S any](ctx context.Context, e *speechEvaluation, kind string, manifest any, definition artifact.ID, count int, attempt func(*speechrecognition.SpeechLease, int, speechrecognition.RunBinding) (speechObservation[S], []artifact.ID, error)) error {
	declaration, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/"+kind+"-evaluation-manifest/v1"), manifest)
	if err != nil {
		return err
	}
	if err := e.publish(ctx, artifact.Batch{Key: kind + "-evaluation/manifest/" + declaration.Descriptor.ID.String(), Contents: []artifact.Content{declaration}}, nil); err != nil {
		return err
	}
	session, err := speechrecognition.LoadSession(ctx, e.store, definition, e.memory)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))
	lease, err := session.Lease(ctx)
	if err != nil {
		return err
	}
	defer lease.Release()
	results := make([]speechObservation[S], count)
	dependencies := []artifact.ID{declaration.Descriptor.ID, definition, e.environment.ID}
	failed := 0
	for i := range count {
		result, parents, err := attempt(lease, i, speechrecognition.RunBinding{Key: fmt.Sprintf("%s-evaluation/run/%s/%d", kind, declaration.Descriptor.ID, i), CodeCommit: e.commit, Environment: e.environment.ID})
		if result.Run.Valid() {
			parents = append(parents, result.Run)
		}
		if result.Attribution.Valid() {
			parents = append(parents, result.Attribution)
		}
		dependencies = append(dependencies, parents...)
		if err != nil {
			result.Failure = err.Error()
			failed++
		}
		results[i] = result
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	report, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/"+kind+"-evaluation-report/v1"), results)
	if err != nil {
		return err
	}
	if err := e.publish(ctx, artifact.Batch{Key: kind + "-evaluation/report/" + report.Descriptor.ID.String(), Contents: []artifact.Content{report}, Lineage: artifact.DependencyLineage(report.Descriptor.ID, dependencies...)}, nil); err != nil {
		return err
	}
	if err := json.NewEncoder(os.Stdout).Encode(struct {
		Report artifact.ID `json:"report"`
		Inputs int         `json:"inputs"`
		Failed int         `json:"failed"`
	}{report.Descriptor.ID, count, failed}); err != nil {
		return err
	}
	if failed != 0 {
		return fmt.Errorf("%s evaluation: %d/%d inputs failed; all attempts retained, no partial pass", kind, failed, count)
	}
	return nil
}
