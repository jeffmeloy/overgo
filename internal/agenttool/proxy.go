package agenttool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const CapabilityProxyName = "capability.proxy"

// EffectAdmission is the shared concrete-effect policy boundary.
type EffectAdmission func(InvocationEffect) error

// CapabilityProxyManual is the fixed provider-visible schema. Catalog growth
// changes returned data, never the proxy's manual identity.
func CapabilityProxyManual() (Manual, error) {
	return NewManual(Manual{
		Name: CapabilityProxyName, Description: "List, inspect, or call one exact active capability.", Effect: EffectInspection,
		Arguments: []Field{
			{Name: "action", Kind: FieldString, Required: true},
			{Name: "query", Kind: FieldString},
			{Name: "limit", Kind: FieldInteger},
			{Name: "manual", Kind: FieldString},
			{Name: "arguments", Kind: FieldObject},
		},
		Transport: Transport{Kind: TransportBuiltin},
	})
}

// RegisterCapabilityProxy binds the fixed proxy to the existing catalog and
// executor. It creates neither a registry nor a dispatch path.
func RegisterCapabilityProxy(executor *Executor, reader artifact.Reader, admit EffectAdmission) error {
	if executor == nil || reader == nil || admit == nil {
		return errors.New("agent tool: capability proxy authority is absent")
	}
	return executor.registerBuiltin(CapabilityProxyName, func(ctx context.Context, raw json.RawMessage) (json.RawMessage, error) {
		var request struct {
			Action    string          `json:"action"`
			Query     string          `json:"query,omitempty"`
			Limit     int             `json:"limit,omitempty"`
			Manual    string          `json:"manual,omitempty"`
			Arguments json.RawMessage `json:"arguments,omitempty"`
		}
		if err := strictjson.DecodeBytes(raw, &request); err != nil {
			return nil, err
		}
		snapshot, err := activeCatalog(ctx, reader)
		if err != nil {
			return nil, err
		}
		switch request.Action {
		case "list":
			if strings.TrimSpace(request.Query) == "" {
				return json.Marshal(snapshot.Entries)
			}
			results, searchErr := snapshot.Search(request.Query, request.Limit)
			if searchErr != nil {
				return nil, searchErr
			}
			return json.Marshal(results)
		case "inspect":
			manual, err := activeProxyManual(ctx, reader, snapshot, request.Manual)
			if err != nil {
				return nil, err
			}
			return manual.Content()
		case "call":
			manual, err := activeProxyManual(ctx, reader, snapshot, request.Manual)
			if err != nil {
				return nil, err
			}
			if manual.Name == CapabilityProxyName {
				return nil, errors.New("agent tool: capability proxy cannot call itself")
			}
			planned, err := DeriveInvocationEffect(manual, request.Arguments, nil)
			if err != nil {
				return nil, err
			}
			if err := admit(planned); err != nil {
				return nil, err
			}
			result, _, err := executor.InvokeWithEffect(ctx, manual, request.Arguments)
			return result, err
		default:
			return nil, errors.New("agent tool: capability proxy action is invalid")
		}
	})
}

func activeCatalog(ctx context.Context, reader artifact.Reader) (CatalogSnapshot, error) {
	id, found, err := reader.ResolveAlias(ctx, ActiveCatalogAlias)
	if err != nil {
		return CatalogSnapshot{}, err
	}
	if !found {
		return CatalogSnapshot{}, errors.New("agent tool: active catalog is absent")
	}
	return RequireCatalogSnapshot(ctx, reader, id)
}

func activeProxyManual(ctx context.Context, reader artifact.Reader, snapshot CatalogSnapshot, text string) (Manual, error) {
	id, err := artifact.ParseID(text)
	if err != nil {
		return Manual{}, errors.New("agent tool: capability proxy requires an exact manual identity")
	}
	entry, found := func() (CatalogEntry, bool) {
		for _, candidate := range snapshot.Entries {
			if candidate.Manual == id {
				return candidate, true
			}
		}
		return CatalogEntry{}, false
	}()
	if !found {
		return Manual{}, errors.New("agent tool: manual is outside the active catalog")
	}
	manual, err := ResolveRegisteredManual(ctx, reader, entry.Name)
	if err != nil {
		return Manual{}, err
	}
	if manual.ID != id || !slices.ContainsFunc(snapshot.Entries, func(item CatalogEntry) bool { return item.Name == manual.Name && item.Manual == manual.ID }) {
		return Manual{}, fmt.Errorf("agent tool: active manual %s changed during dispatch", id)
	}
	return manual, nil
}
