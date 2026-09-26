package gate

import (
	"slices"
	"strings"
	"testing"
)

// TestGateNotesCarryTheirKind holds every audit reader to the kind its
// producer declared, never to the note's words: a warning without a warning
// word prints as one, a detail that happens to say "unavailable" does not,
// and only measured kinds enter the commit message.
func TestGateNotesCarryTheirKind(t *testing.T) {
	t.Parallel()
	g := &gateContext{}
	g.advise(noteWarning, "store admission lost its writer")
	g.note("fixture cache unavailable; recomputed")
	g.advise(noteClass, "operation class: code-change")
	g.advise(noteRepair, "staged repair: gofmt for the fmt phase")
	if got := compactAudit(g.audit); !slices.Equal(got, []string{
		"advisory: warning: store admission lost its writer", "advisory: class: operation class: code-change",
	}) {
		t.Fatalf("compact audit = %q", got)
	}
	message, err := structuredMessage([]byte("Cause: x\n\nPredicted effect: y"), g.audit)
	if err != nil || !strings.HasSuffix(string(message), "Measured by the gate:\n  operation class: code-change\n") {
		t.Fatalf("commit message = %q, %v", message, err)
	}
}
