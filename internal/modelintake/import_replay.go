package modelintake

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/hfgguf"
	"overgo/internal/strictjson"
)

// ReplayImport reconstructs a saved inspection only when every canonical fact matches.
//   - Physical locations may change while source and mapping identities remain fixed.
//   - A mixed output contract uses the same per-class defaults as inspection.
//   - Publication and numerical verification remain separate operations.
func ReplayImport(ctx context.Context, path string, data []byte) (ImportInspection, error) {
	if ctx == nil {
		return ImportInspection{}, errors.New("model intake: import replay requires a context")
	}
	if err := ctx.Err(); err != nil {
		return ImportInspection{}, err
	}
	if len(data) == 0 || len(data) > artifact.MaxContentBytes {
		return ImportInspection{}, errors.New("model intake: saved inspection exceeds the content size contract")
	}
	var description importDescription
	if err := strictjson.DecodeBytes(data, &description); err != nil {
		return ImportInspection{}, err
	}
	if description.Version != artifact.InitialDocumentVersion || description.Qualification != "structural-only" || len(description.Components) == 0 {
		return ImportInspection{}, errors.New("model intake: saved inspection requires the current structural component contract")
	}
	frozen, err := artifact.JSONContent(importContract, description)
	if err != nil {
		return ImportInspection{}, err
	}
	purposes := map[hfgguf.LanguageOutput]bool{}
	for _, component := range description.Components {
		if component.Purpose != "" {
			purposes[component.Purpose] = true
		}
	}
	var output hfgguf.LanguageOutput
	if len(purposes) == 1 {
		for purpose := range purposes {
			output = purpose
		}
	}
	current, err := InspectImport(ctx, path, output)
	if err != nil {
		return ImportInspection{}, err
	}
	if current.Content.Descriptor.ID != frozen.Descriptor.ID {
		return ImportInspection{}, fmt.Errorf("model intake: sources or mappings differ from saved inspection %s; current inspection is %s", frozen.Descriptor.ID, current.Content.Descriptor.ID)
	}
	return current, nil
}
