package testevidence

import (
	"bytes"
	"os"
	"os/exec"
	"slices"
	"testing"

	"overgo/internal/testskip"
)

const typedOutcomeChild = "OVERGO_TEST_TYPED_OUTCOME_CHILD"

// TestOutcomesClassifyFromTypedEvents holds skip classification to the kind
// a testskip call records, carried through test2json as an attr event: a
// short exclusion is classified in a short run only, even when its message
// quotes the inapplicable marker, and a declared inapplicability in every
// run. Only a test that recorded no kind -- a frozen harness, a stored
// receipt -- is read by its message, and a skip citing neither stays unowned.
func TestOutcomesClassifyFromTypedEvents(t *testing.T) {
	if os.Getenv(typedOutcomeChild) != "" {
		t.Run("Short", func(t *testing.T) { testskip.Short(t, testskip.Inapplicable+" is quoted, not recorded") })
		t.Run("Inapplicable", func(t *testing.T) { testskip.NotApplicable(t, "another lane") })
		t.Run("Frozen", func(t *testing.T) { t.Skip(testskip.Inapplicable + ": a frozen harness") })
		t.Run("Unowned", func(t *testing.T) { t.Skip("fixture missing") })
		return
	}
	t.Parallel()
	command := exec.CommandContext(t.Context(), "go", "tool", "test2json", "-p", "example", os.Args[0],
		"-test.v=test2json", "-test.short", "-test.run=^TestOutcomesClassifyFromTypedEvents$")
	command.Env = append(os.Environ(), typedOutcomeChild+"=1")
	var stream bytes.Buffer
	command.Stdout, command.Stderr = &stream, &stream
	if err := command.Run(); err != nil {
		t.Fatalf("child run: %v\n%s", err, stream.String())
	}
	name := func(subtest string) string { return "example: TestOutcomesClassifyFromTypedEvents/" + subtest }
	for short, want := range map[bool][2][]string{
		true:  {{name("Short"), name("Inapplicable"), name("Frozen")}, {name("Unowned")}},
		false: {{name("Inapplicable"), name("Frozen")}, {name("Short"), name("Unowned")}},
	} {
		report, err := readGoTestJSON(bytes.NewReader(stream.Bytes()), short, false, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(report.ClassifiedSkipped, want[0]) || !slices.Equal(report.Skipped, want[1]) {
			t.Fatalf("short=%t classified %v unowned %v, want %v", short, report.ClassifiedSkipped, report.Skipped, want)
		}
	}
}
