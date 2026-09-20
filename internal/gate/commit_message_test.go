package gate

import (
	"strings"
	"testing"
)

// TestStructuredCommitMessage holds a landing message to its two halves. The
// authored half must say why the change exists and what it should do, and a
// message missing either section is refused by name. The measured half is the
// gate's: it carries the notes the gate measured, in the order it measured
// them, below the authored text and apart from it, and none of the notes that
// only advise; a gate that measured nothing appends nothing.
func TestStructuredCommitMessage(t *testing.T) {
	t.Parallel()
	const authored = "Subject line\n\nCause: the store opened twice.\nPredicted effect: one open.\n"
	for section, message := range map[string]string{
		"Cause:":            "Subject\n\nPredicted effect: one open.\n",
		"Predicted effect:": "Subject\n\nCause: the store opened twice.\n",
	} {
		if _, err := structuredMessage([]byte(message), nil); err == nil || !strings.Contains(err.Error(), section) {
			t.Fatalf("a message without %q = %v", section, err)
		}
	}
	plain, err := structuredMessage([]byte(authored), nil)
	if err != nil || string(plain) != authored {
		t.Fatalf("a gate that measured nothing wrote %q, %v", plain, err)
	}
	audit := []string{
		"impact selection: claims triggered: authority:compatibility",
		"operation class: code-change",
		"test scope: 195 opaque reader(s) bound to every root",
		"code profile delta vs HEAD: runtime=+0 files/+19 nodes",
		"automation ROI commit: production_ast=0-4 net=-4",
	}
	got, err := structuredMessage([]byte(authored), audit)
	want := strings.TrimSpace(authored) + "\n\nMeasured by the gate:\n  operation class: code-change\n" +
		"  code profile delta vs HEAD: runtime=+0 files/+19 nodes\n  automation ROI commit: production_ast=0-4 net=-4\n"
	if err != nil || string(got) != want {
		t.Fatalf("structured message = %q, %v; want %q", got, err, want)
	}
	if len(audit) != 5 || audit[0] != "impact selection: claims triggered: authority:compatibility" {
		t.Fatalf("building the message rewrote the gate's audit: %q", audit)
	}
}
