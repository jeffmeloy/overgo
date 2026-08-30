package agenttool

import "testing"

// TestRSIRuntimeIsGoOnly pins the transport closed world: every tool
// invocation rides one of the compiled Go adapters, an interpreter or script
// transport does not exist, and an unregistered transport kind resolves to
// nothing. Manuals stay typed data; only this compiled set executes them.
func TestRSIRuntimeIsGoOnly(t *testing.T) {
	executor := NewExecutor()
	compiled := map[TransportKind]bool{
		TransportBuiltin:        true,
		TransportHTTP:           true,
		TransportArgv:           true,
		TransportMCPHTTP:        true,
		TransportHTTPJSONStream: true,
	}
	if len(executor.adapters) != len(compiled) {
		t.Fatalf("transport adapter set has %d entries, want %d", len(executor.adapters), len(compiled))
	}
	for kind := range executor.adapters {
		if !compiled[kind] {
			t.Fatalf("transport %q is outside the compiled Go adapter set", kind)
		}
	}
	for _, rogue := range []TransportKind{"script", "shell", "python", "plugin", "mcp-stdio"} {
		if _, registered := executor.adapters[rogue]; registered {
			t.Fatalf("interpreter transport %q is registered", rogue)
		}
	}
}
