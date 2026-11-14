package inferencesurface

import (
	"slices"
	"testing"
)

// TestAResultReadIsFoundByItsImportedOwner holds a package's result settings to the reads its code makes of them
// (SQA-381): a read was found by its spelling, envsetting.Result, so a file that imports the package under another name
// hid its read from the identity, and a stale result read as current. A file importing it as settings, beside a comment
// spelling envsetting.Result and a local of the import's own name with a Result of its own, reads the aliased setting
// alone; Result taken as a value, and Result given a name no count can read, are reads it cannot count.
func TestAResultReadIsFoundByItsImportedOwner(t *testing.T) {
	t.Parallel()
	aliased := `package fixture

import settings "overgo/internal/envsetting"

// envsetting.Result("COMMENTED") names a read in a comment.
func read() string { return settings.Result("ALIASED") }

func shadowed() string {
	settings := lookup{}
	return settings.Result("SHADOWED")
}

type lookup struct{}

func (lookup) Result(name string) string { return name }
`
	if results, other := environmentReads("aliased.go", []byte(aliased)); !slices.Equal(results, []string{"ALIASED"}) || len(other) != 0 {
		t.Errorf("the aliased file reads %v and cannot count %v; want ALIASED alone", results, other)
	}
	dynamic := `package fixture

import "overgo/internal/envsetting"

var read = envsetting.Result

func named(name string) string { return envsetting.Result(name) }
`
	if results, other := environmentReads("dynamic.go", []byte(dynamic)); len(results) != 0 || len(other) != 2 {
		t.Errorf("Result as a value and with a computed name read %v and cannot count %v; want two reads no count can follow", results, other)
	}
}
