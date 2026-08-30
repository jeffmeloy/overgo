package agenttool

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/url"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/strictjson"
)

// InvocationTarget retains the agenttool target API while its canonical
// transport-neutral definition lives in invocation.
type InvocationTarget = invocation.Target

// InvocationEffect retains the agenttool effect API while its canonical
// transport-neutral definition lives in invocation.
type InvocationEffect = invocation.Effect

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
	effect := invocation.Effect{
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
		effect.Targets = append(effect.Targets, invocation.Target{Scope: binding.Scope, Value: value})
	}
	switch manual.Transport.Kind {
	case TransportHTTP, TransportMCPHTTP, TransportHTTPJSONStream:
		effect.Network = true
		endpoint, parseErr := url.Parse(manual.Transport.URL)
		if parseErr != nil {
			return InvocationEffect{}, parseErr
		}
		effect.Targets = append(effect.Targets, invocation.Target{Scope: EffectScopeExternal, Value: endpoint.Scheme + "://" + endpoint.Host})
	case TransportArgv:
		effect.Executable = true
	}
	if manual.Effect == EffectMutation {
		// A fixed argv program remains executable, but an exact declared target
		// still makes its world effect inspectable. Missing targets stay opaque.
		effect.Known = len(effect.Targets) != 0
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
	return invocation.NewEffect(effect)
}

// CanonicalArguments validates a manual payload and returns the exact bytes
// whose identity DeriveInvocationEffect binds.
func CanonicalArguments(manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	if err := validateArguments(manual, arguments); err != nil {
		return nil, err
	}
	canonical, err := canonicalObject(arguments)
	return json.RawMessage(canonical), err
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
