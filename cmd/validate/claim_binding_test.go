package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestATestFileClaimMovesWhenTheFilesInitChanges holds a claim's test-file evidence that names no symbol to the whole
// file (SQA-316): the claim bound a same-file graph rooted at the declarations that held its text, so a func init that
// set a variable the named test reads changed what the test observes while the claim kept its identity. Changing the
// fixture's init assignment from 3 to 4 moves the bound bytes; line ends written as a checkout writes them do not.
func TestATestFileClaimMovesWhenTheFilesInitChanges(t *testing.T) {
	t.Parallel()
	const file = "package p\n\nimport \"testing\"\n\nvar limit int\n\nfunc init() { limit = 3 }\n\n" +
		"func TestNamed(t *testing.T) {\n\tif limit != 3 {\n\t\tt.Fatal(limit)\n\t}\n}\n"
	proof := evidence{Path: "p/p_test.go", Contains: "func TestNamed(", Role: roleSource}
	bound := func(source string) []byte {
		t.Helper()
		return testcheck.Must(evidencePayload(proof, []byte(source)))(t)
	}
	original := bound(file)
	if bytes.Equal(bound(strings.Replace(file, "limit = 3", "limit = 4", 1)), original) {
		t.Fatal("a change to the file's init left the claim's bound bytes as they were")
	}
	if !bytes.Equal(bound(strings.ReplaceAll(file, "\n", "\r\n")), original) {
		t.Fatal("line ends written as a checkout writes them moved the claim's bound bytes")
	}
}
