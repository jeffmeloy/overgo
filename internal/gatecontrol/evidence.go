package gatecontrol

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

// OutstandingDebt derives prepared-but-not-finalized gate work from RepoDB.
func (store Store) OutstandingDebt(ctx context.Context) ([]runrecord.GateLifecycle, error) {
	repository, err := repodb.OpenReadOnly(filepath.Join(store.root, store.repository))
	if err != nil {
		return nil, err
	}
	defer repository.Close()
	result, err := repository.Query(ctx, repodb.Query{
		Kind: artifact.KindEvidence, MaxResults: repodb.MaxQueryResults,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, errors.New("gate control: RepoDB evidence query was truncated")
	}
	contents := make([]artifact.Content, 0)
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != runrecord.GateLifecycleMediaType || descriptor.Schema != runrecord.GateLifecycleSchema {
			continue
		}
		content, ok, err := repository.Content(ctx, descriptor.ID)
		if err != nil {
			return nil, err
		}
		if ok {
			contents = append(contents, content)
		}
	}
	return runrecord.OutstandingGateDebt(contents)
}
