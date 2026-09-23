package codeprofile

// EvidenceSchema names the per-gate code-profile documents the gate wrote
// until 2026-09-23. Nothing read them back -- each gate measures its own
// candidate and base -- so the gate no longer writes them; the schema stays
// so the store's release still finds the ones already held.
const EvidenceSchema = "overgo/code-profile/v4"

func hasFunctionPair(functions []string) bool {
	found := false
	for range functions {
		if found {
			return true
		}
		found = true
	}
	return false
}
