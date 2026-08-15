package codetighten

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

func TestExactForwarderReduction(t *testing.T) {
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module fixture\n\ngo 1.26\n")
	testutil.WriteTextFile(t, root, "value/value.go", `package value

func add(a, b int) int { return a + b }
func sum(a, b int) int { return add(a, b) }
func Value() int { return sum(1, 2) }
`)
	testutil.WriteTextFile(t, root, "value/other.go", `package value

func other() int { return sum(2, 3) }
`)
	testutil.WriteTextFile(t, root, "value/value_test.go", `package value

import "testing"

func TestValue(t *testing.T) {
	if Value() != 3 || other() != 5 { t.Fatal("behavior changed") }
}
`)

	proposals, err := Discover(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(proposals) != 1 || proposals[0].ID != "value:sum->add" || proposals[0].Calls != 2 {
		t.Fatalf("proposals = %+v", proposals)
	}
	if _, err := Apply(root, proposals[0].ID); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"value/value.go", "value/other.go"} {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), "sum(") {
			t.Fatalf("%s retains displaced wrapper or caller:\n%s", name, data)
		}
	}

	t.Run("refuses function values", func(t *testing.T) {
		testutil.WriteTextFile(t, root, "value/alias.go", `package value

func plus(a, b int) int { return add(a, b) }
var operation = plus
`)
		proposals, err := Discover(root)
		if err != nil {
			t.Fatal(err)
		}
		for _, proposal := range proposals {
			if proposal.Wrapper == "plus" {
				t.Fatalf("unsafe function-value reduction proposed: %+v", proposal)
			}
		}
	})

	t.Run("ignores non-identifier calls", func(t *testing.T) {
		testutil.WriteTextFile(t, root, "value/selector.go", `package value

import "fmt"

func rendered() string { return fmt.Sprint(1) }
`)
		if _, err := Discover(root); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("rolls back failed behavior check", func(t *testing.T) {
		testutil.WriteTextFile(t, root, "stack/stack.go", `package stack

import "runtime"

func caller() string {
	pc, _, _, _ := runtime.Caller(1)
	return runtime.FuncForPC(pc).Name()
}
func wrapper() string { return caller() }
func Value() string { return wrapper() }
`)
		testutil.WriteTextFile(t, root, "stack/stack_test.go", `package stack

import (
	"strings"
	"testing"
)

func TestStack(t *testing.T) {
	if !strings.HasSuffix(Value(), ".wrapper") { t.Fatal(Value()) }
}
`)
		before, err := os.ReadFile(filepath.Join(root, "stack", "stack.go"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Apply(root, "stack:wrapper->caller"); err == nil {
			t.Fatal("stack-sensitive behavior change passed")
		}
		after, err := os.ReadFile(filepath.Join(root, "stack", "stack.go"))
		if err != nil {
			t.Fatal(err)
		}
		if string(after) != string(before) {
			t.Fatalf("failed transaction was not rolled back:\n%s", after)
		}
	})
}
