package agenttool

import (
	"encoding/json"
	"testing"
)

func TestInvocationEffectDerivation(t *testing.T) {
	manual := validManual()
	manual.Effect = EffectMutation
	manual.Ceiling = EffectCeiling{
		Targets:     []EffectTargetBinding{{Scope: EffectScopeWorkspace, Argument: "pattern"}},
		Destructive: true,
	}
	identified, err := NewManual(manual)
	if err != nil {
		t.Fatal(err)
	}
	first, err := DeriveInvocationEffect(identified, json.RawMessage(`{"limit":2,"pattern":"internal/agenttool"}`), json.RawMessage(`{"ok":true}`))
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveInvocationEffect(identified, json.RawMessage(`{"pattern":"internal/agenttool","limit":2}`), json.RawMessage(`{"ok":true}`))
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || !first.Known || first.OpaqueMutation || !first.Destructive || len(first.Targets) != 1 {
		t.Fatalf("resolved effect = %+v; reordered = %+v", first, second)
	}

	remote := validManual()
	remote.Transport = Transport{Kind: TransportHTTP, URL: "https://example.test/tool"}
	remote, err = NewManual(remote)
	if err != nil {
		t.Fatal(err)
	}
	inspection, err := DeriveInvocationEffect(remote, json.RawMessage(`{"pattern":"x"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !inspection.Known || !inspection.Network || inspection.Class != EffectInspection || inspection.Targets[0].Scope != EffectScopeExternal {
		t.Fatalf("remote inspection = %+v", inspection)
	}

	opaque := validManual()
	opaque.Effect = EffectMutation
	opaque.Transport = Transport{Kind: TransportArgv, Program: "git"}
	opaque, err = NewManual(opaque)
	if err != nil {
		t.Fatal(err)
	}
	unknown, err := DeriveInvocationEffect(opaque, json.RawMessage(`{"pattern":"x"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if unknown.Known || !unknown.OpaqueMutation || !unknown.Executable {
		t.Fatalf("argv mutation = %+v", unknown)
	}
}
