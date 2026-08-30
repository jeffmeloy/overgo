package agenttool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// StandardManuals declares the store-inspection builtins every agent
// runtime carries: reading the catalog is the inspection ground the
// mutation gate stands on.
func StandardManuals() ([]Manual, error) {
	declarations := []Manual{
		{
			Name:        "store.head",
			Description: "Report the catalog head commit and sequence.",
			Effect:      EffectInspection,
			Ceiling:     EffectCeiling{Targets: []EffectTargetBinding{{Scope: EffectScopeRepository, Value: "overgodb"}}},
			Transport:   Transport{Kind: TransportBuiltin},
		},
		{
			Name:        "store.alias",
			Description: "Resolve one catalog alias to its artifact identity.",
			Effect:      EffectInspection,
			Ceiling:     EffectCeiling{Targets: []EffectTargetBinding{{Scope: EffectScopeRepository, Value: "overgodb"}}},
			Arguments: []Field{
				{Name: "name", Kind: FieldString, Required: true, Description: "exact alias"},
			},
			Transport: Transport{Kind: TransportBuiltin},
		},
	}
	manuals := make([]Manual, len(declarations))
	for index, declaration := range declarations {
		manual, err := NewManual(declaration)
		if err != nil {
			return nil, err
		}
		manuals[index] = manual
	}
	return manuals, nil
}

// HeadReader is the store surface the standard builtins inspect.
type HeadReader interface {
	artifact.Reader
	Head() (artifact.CommitID, uint64)
}

// RegisterStandardBuiltins binds the store-inspection builtins to the
// executor over the supplied read-only catalog.
func RegisterStandardBuiltins(executor *Executor, store HeadReader) error {
	if executor == nil || store == nil {
		return errors.New("agent tool: nil executor or store for standard builtins")
	}
	if err := executor.registerBuiltin("store.head", func(context.Context, json.RawMessage) (json.RawMessage, error) {
		head, sequence := store.Head()
		return json.Marshal(map[string]any{"head": head, "sequence": sequence})
	}); err != nil {
		return err
	}
	return executor.registerBuiltin("store.alias", func(ctx context.Context, arguments json.RawMessage) (json.RawMessage, error) {
		var request struct {
			Name string `json:"name"`
		}
		if err := strictjson.DecodeBytes(arguments, &request); err != nil {
			return nil, err
		}
		target, found, err := store.ResolveAlias(ctx, request.Name)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, fmt.Errorf("alias %q is not bound", request.Name)
		}
		return json.Marshal(map[string]any{"name": request.Name, "target": target})
	})
}
