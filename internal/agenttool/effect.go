package agenttool

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/url"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// InvocationTarget is one resolved authority boundary touched by a call.
type InvocationTarget struct {
	Scope EffectScope `json:"scope"`
	Value string      `json:"value"`
}

// InvocationEffect is the canonical, concrete effect shared by admission,
// execution evidence, trajectories, and policy. Unknown writers are opaque.
type InvocationEffect struct {
	ID             artifact.ID        `json:"-"`
	Manual         artifact.ID        `json:"manual"`
	Arguments      artifact.ID        `json:"arguments"`
	ObservedResult *artifact.ID       `json:"observed_result,omitempty"`
	Class          Effect             `json:"class"`
	Targets        []InvocationTarget `json:"targets,omitempty"`
	Known          bool               `json:"known"`
	OpaqueMutation bool               `json:"opaque_mutation,omitzero"`
	Network        bool               `json:"network,omitzero"`
	Executable     bool               `json:"executable,omitzero"`
	Destructive    bool               `json:"destructive,omitzero"`
	Privileged     bool               `json:"privileged,omitzero"`
	Irreversible   bool               `json:"irreversible,omitzero"`
}

// DeriveInvocationEffect validates and canonicalizes a concrete invocation.
// observedResult is optional before dispatch and binds the same effect record
// to the result afterward.
func DeriveInvocationEffect(manual Manual, arguments, observedResult json.RawMessage) (InvocationEffect, error) {
	if !manual.ID.Valid() {
		return InvocationEffect{}, errors.New("agent tool: invocation effect requires an identified manual")
	}
	if err := validateArguments(manual, arguments); err != nil {
		return InvocationEffect{}, err
	}
	canonicalArguments, err := canonicalObject(arguments)
	if err != nil {
		return InvocationEffect{}, err
	}
	argumentID, err := artifact.IdentifyBytes(artifact.KindEvidence, canonicalArguments)
	if err != nil {
		return InvocationEffect{}, err
	}
	effect := InvocationEffect{
		Manual: manual.ID, Arguments: argumentID, Class: manual.Effect,
		Known:       manual.Effect == EffectInspection,
		Destructive: manual.Ceiling.Destructive, Privileged: manual.Ceiling.Privileged,
		Irreversible: manual.Ceiling.Irreversible,
	}
	var supplied map[string]json.RawMessage
	if err := strictjson.DecodeBytes(canonicalArguments, &supplied); err != nil {
		return InvocationEffect{}, err
	}
	for _, binding := range manual.Ceiling.Targets {
		value := binding.Value
		if binding.Argument != "" {
			raw, found := supplied[binding.Argument]
			if !found || json.Unmarshal(raw, &value) != nil || strings.TrimSpace(value) == "" {
				return InvocationEffect{}, errors.New("agent tool: invocation target cannot be resolved")
			}
		}
		effect.Targets = append(effect.Targets, InvocationTarget{Scope: binding.Scope, Value: value})
	}
	switch manual.Transport.Kind {
	case TransportHTTP, TransportMCPHTTP, TransportHTTPJSONStream:
		effect.Network = true
		endpoint, parseErr := url.Parse(manual.Transport.URL)
		if parseErr != nil {
			return InvocationEffect{}, parseErr
		}
		effect.Targets = append(effect.Targets, InvocationTarget{Scope: EffectScopeExternal, Value: endpoint.Scheme + "://" + endpoint.Host})
	case TransportArgv:
		effect.Executable = true
	}
	slices.SortFunc(effect.Targets, func(a, b InvocationTarget) int {
		if order := strings.Compare(string(a.Scope), string(b.Scope)); order != 0 {
			return order
		}
		return strings.Compare(a.Value, b.Value)
	})
	effect.Targets = slices.Compact(effect.Targets)
	if manual.Effect == EffectMutation {
		effect.Known = len(effect.Targets) != 0 && manual.Transport.Kind != TransportArgv
		effect.OpaqueMutation = !effect.Known
	}
	if len(observedResult) != 0 {
		result, boundedErr := boundedResult(manual.Name, observedResult)
		if boundedErr != nil {
			return InvocationEffect{}, boundedErr
		}
		resultID, identifyErr := artifact.IdentifyBytes(artifact.KindEvidence, result)
		err = identifyErr
		if err != nil {
			return InvocationEffect{}, err
		}
		effect.ObservedResult = &resultID
	}
	effect.ID, err = artifact.JSONID(artifact.KindEvidence, effect)
	return effect, err
}

func canonicalObject(data []byte) ([]byte, error) {
	var value map[string]json.RawMessage
	if err := strictjson.DecodeBytes(data, &value); err != nil || value == nil {
		return nil, errors.New("agent tool: arguments are not a strict JSON object")
	}
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSpace(buffer.Bytes()), nil
}
