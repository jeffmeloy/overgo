package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAClaimBindsTheTestItNames holds a claim's test-file evidence to the test it names and what that test runs in
// its file: a change to another test of the file, to an import only that test uses or to a comment leaves the bound
// bytes as they were, and a change to the named test, to a helper or constant of the file it calls, or to an import
// it uses moves them. A production file's evidence still binds the whole file.
func TestAClaimBindsTheTestItNames(t *testing.T) {
	t.Parallel()
	const file = `package p

import (
	"strings"
	"testing"
)

const limit = 3

func helper() int { return limit }

// namedCheck is the test a claim names.
func namedCheck(t *testing.T) {
	if helper() != 3 {
		t.Fatal("helper")
	}
}

func otherCheck(t *testing.T) {
	_ = strings.ToUpper("other")
}
`
	proof := evidence{Path: "p/p_test.go", Contains: "func namedCheck(", Role: roleSource}
	bound := func(source string) []byte {
		t.Helper()
		return testcheck.Must(evidencePayload(proof, []byte(source)))(t)
	}
	original := bound(file)
	for _, unchanged := range []struct{ change, from, to string }{
		{"another test's body", `strings.ToUpper("other")`, `strings.ToLower("other")`},
		{"an import only another test uses", `"strings"`, `strings "strings"`},
		{"the named test's comment", "the test a claim names", "the test its claim names"},
	} {
		if got := bound(strings.Replace(file, unchanged.from, unchanged.to, 1)); !bytes.Equal(got, original) {
			t.Errorf("a change to %s moved the bound bytes", unchanged.change)
		}
	}
	for _, moved := range []struct{ change, from, to string }{
		{"the named test", `t.Fatal("helper")`, `t.Fatal("helper differs")`},
		{"a helper it calls", "return limit", "return limit + 0"},
		{"a constant its helper reads", "const limit = 3", "const limit = 4"},
		{"an import it uses", `"testing"`, `testing "testing"`},
	} {
		if got := bound(strings.Replace(file, moved.from, moved.to, 1)); bytes.Equal(got, original) {
			t.Errorf("a change to %s left the bound bytes as they were", moved.change)
		}
	}
	production := evidence{Path: "p/p.go", Contains: "func helper(", Role: roleSource}
	whole := testcheck.Must(evidencePayload(production, []byte(file)))(t)
	if !bytes.Equal(whole, canonicalEvidence([]byte(file))) {
		t.Error("a production file's evidence no longer binds the whole file")
	}
}
