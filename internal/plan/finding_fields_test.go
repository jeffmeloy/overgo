package plan

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAFindingHoldsItsSixFields holds the review's register to the owner's rule of 2026-11-14
// (single-source/finding-fields): a finding held up to sixteen fields, its status, title, evidence and closure criteria
// among them, and the plan read four. A finding holds its id, severity, the files it cites, the input that shows it,
// what is expected of that input and what actually happens, and is open while the register holds it. The reader's type
// holds those six alone, a finding naming its status is refused, and each finding the register holds gives all six.
func TestAFindingHoldsItsSixFields(t *testing.T) {
	t.Parallel()
	read := reflect.TypeFor[openFinding]()
	fields := make([]string, read.NumField())
	for index := range fields {
		fields[index], _, _ = strings.Cut(read.Field(index).Tag.Get("json"), ",")
	}
	if want := []string{"id", "severity", "files", "input", "expected", "actual"}; !slices.Equal(fields, want) {
		t.Fatalf("a finding is read as %v, want %v", fields, want)
	}
	if _, err := readFindings([]byte(`{"findings": [{"id": "SQA-1", "severity": "P2", "files": ["p/p.go"], "input": "i",
		"expected": "e", "actual": "a", "status": "open"}]}`)); err == nil {
		t.Fatal("a finding naming its status was read")
	}
	register := testcheck.Must(os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(FindingsRegister))))(t)
	for _, finding := range testcheck.Must(readFindings(register))(t) {
		value := reflect.ValueOf(finding)
		for index, field := range fields {
			if value.Field(index).IsZero() {
				t.Errorf("%s %s gives no %s", FindingsRegister, finding.ID, field)
			}
		}
	}
}
