package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestConsumerReducerRetainsOnlyVerifiedReduction(t *testing.T) {
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module fixture\n\ngo 1.26\n")
	testutil.WriteTextFile(t, root, "value/api.go", `package value

func Local(v int) int { return v + 1 }
func Value() int { return Local(2) + keep() }
func unusedAPI() {}
`)
	testutil.WriteTextFile(t, root, "value/mixed.go", `package value

import "fmt"

func unusedFormat() string { return fmt.Sprint("unused") }
func keep() int { return 1 }
`)
	testutil.WriteTextFile(t, root, "value/empty.go", "package value\nfunc unusedEmpty() {}\n")
	testutil.WriteTextFile(t, root, "value/api_test.go", `package value

import "testing"

func TestValue(t *testing.T) {
	if Value() != 4 { t.Fatal(Value()) }
}
`)

	proposals, err := discover(root)
	if err != nil {
		t.Fatal(err)
	}
	proposal := findProposal(proposals, "value:consumer-surface")
	if proposal == nil || !strings.Contains(strings.Join(proposal.Symbols, ","), "privatize:Local->local") {
		t.Fatalf("consumer proposal = %+v", proposal)
	}
	if _, err := apply(root, proposal.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "value", "empty.go")); !os.IsNotExist(err) {
		t.Fatalf("empty source file retained: %v", err)
	}
	for _, name := range []string{"api.go", "mixed.go"} {
		data, err := os.ReadFile(filepath.Join(root, "value", name))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.Contains(text, "Local") || strings.Contains(text, "unusedFormat") || strings.Contains(text, `"fmt"`) {
			t.Fatalf("%s retains displaced surface:\n%s", name, text)
		}
	}

	testutil.WriteTextFile(t, root, "guard/guard.go", "package guard\nfunc hidden() {}\n")
	testutil.WriteTextFile(t, root, "guard/guard_test.go", `package guard

import (
	"os"
	"strings"
	"testing"
)

func TestSourceContract(t *testing.T) {
	data, err := os.ReadFile("guard.go")
	if err != nil || !strings.Contains(string(data), "hidden") { t.Fatal("source contract changed") }
}
`)
	before, _ := os.ReadFile(filepath.Join(root, "guard", "guard.go"))
	proposals, err = discover(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := apply(root, findProposal(proposals, "guard:consumer-surface").ID); err == nil {
		t.Fatal("failed package verification retained a reduction")
	}
	after, err := os.ReadFile(filepath.Join(root, "guard", "guard.go"))
	if err != nil || string(after) != string(before) {
		t.Fatalf("failed transaction was not restored: %v", err)
	}
}

func findProposal(proposals []candidate, id string) *candidate {
	for index := range proposals {
		if proposals[index].ID == id {
			return &proposals[index]
		}
	}
	return nil
}
