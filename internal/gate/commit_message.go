package gate

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// structuredMessage holds the authored message to its sections and appends
// what the gate measured (its measured notes, the numbers it computed). The
// author says why the change exists and what it should do; a number belongs
// in the block below it, so a prediction is never mistaken for a measurement.
// Admission passes no audit and only checks.
func structuredMessage(authored []byte, audit []auditNote) ([]byte, error) {
	text := strings.TrimSpace(string(authored))
	for _, section := range []string{"Cause:", "Predicted effect:"} {
		if !strings.Contains(text, section) {
			return nil, fmt.Errorf("commit message: no %q section; state why the change exists and what it should do, and leave what was measured to the gate", section)
		}
	}
	var measured []string
	for _, note := range audit {
		if note.kind >= noteSuiteCost && note.kind <= noteDebt {
			measured = append(measured, note.text)
		}
	}
	if len(measured) != 0 {
		text += "\n\nMeasured by the gate:\n  " + strings.Join(measured, "\n  ")
	}
	return []byte(text + "\n"), nil
}

// standardInput names the gate's own input as the message file.
const standardInput = "-"

// authoredMessage reads the operator's message, once: standard input cannot
// be read again at the commit, so the gate keeps what admission read.
func authoredMessage(path string, input io.Reader) ([]byte, error) {
	if path == standardInput {
		return io.ReadAll(input)
	}
	return os.ReadFile(path)
}
