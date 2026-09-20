package gate

import (
	"fmt"
	"slices"
	"strings"
)

// measuredNotes are the audit notes that state what the gate measured about
// a landing, as opposed to what it advised: they are the numbers a landing
// message may carry, written by the gate that computed them.
var measuredNotes = []string{
	"operation class:", "code profile delta vs HEAD", "automation ROI", "harness surface raised", "suite cost ranking:",
}

// structuredMessage holds the authored message to its sections and appends
// what the gate measured. The author says why the change exists and what it
// should do; a number belongs in the block below it, so a prediction is never
// mistaken for a measurement. Admission passes no audit and only checks.
func structuredMessage(authored []byte, audit []string) ([]byte, error) {
	text := strings.TrimSpace(string(authored))
	for _, section := range []string{"Cause:", "Predicted effect:"} {
		if !strings.Contains(text, section) {
			return nil, fmt.Errorf("commit message: no %q section; state why the change exists and what it should do, and leave what was measured to the gate", section)
		}
	}
	measured := slices.DeleteFunc(slices.Clone(audit), func(note string) bool {
		return !slices.ContainsFunc(measuredNotes, func(prefix string) bool { return strings.HasPrefix(note, prefix) })
	})
	if len(measured) != 0 {
		text += "\n\nMeasured by the gate:\n  " + strings.Join(measured, "\n  ")
	}
	return []byte(text + "\n"), nil
}
