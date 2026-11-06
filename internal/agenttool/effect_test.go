package agenttool

import (
	"encoding/json"
	"testing"

	"overgo/internal/testcheck"
)

func TestInvocationEffectDerivation(t *testing.T) {
	manual := validManual()
	manual.Effect = EffectMutation
	manual.Ceiling = EffectCeiling{
		Targets:     []EffectTargetBinding{{Scope: EffectScopeWorkspace, Argument: "pattern"}},
		Destructive: true,
	}
	identified := testcheck.Must(NewManual(manual))(t)
	first := testcheck.Must(DeriveInvocationEffect(identified, json.RawMessage(`{"limit":2,"pattern":"internal/agenttool"}`), json.RawMessage(`{"ok":true}`)))(t)
	second := testcheck.Must(DeriveInvocationEffect(identified, json.RawMessage(`{"pattern":"internal/agenttool","limit":2}`), json.RawMessage(`{"ok":true}`)))(t)
	if first.ID != second.ID || !first.Known || first.OpaqueMutation || !first.Destructive || len(first.Targets) != 1 {
		t.Fatalf("resolved effect = %+v; reordered = %+v", first, second)
	}

	remote := validManual()
	remote.Transport = Transport{Kind: TransportHTTP, URL: "https://example.test/tool"}
	remote = testcheck.Must(NewManual(remote))(t)
	inspection := testcheck.Must(DeriveInvocationEffect(remote, json.RawMessage(`{"pattern":"x"}`), nil))(t)
	if !inspection.Known || !inspection.Network || inspection.Class != EffectInspection || inspection.Targets[0].Scope != EffectScopeExternal {
		t.Fatalf("remote inspection = %+v", inspection)
	}

	opaque := validManual()
	opaque.Effect = EffectMutation
	opaque.Transport = Transport{Kind: TransportArgv, Program: "git"}
	opaque = testcheck.Must(NewManual(opaque))(t)
	unknown := testcheck.Must(DeriveInvocationEffect(opaque, json.RawMessage(`{"pattern":"x"}`), nil))(t)
	if unknown.Known || !unknown.OpaqueMutation || !unknown.Executable {
		t.Fatalf("argv mutation = %+v", unknown)
	}
}
