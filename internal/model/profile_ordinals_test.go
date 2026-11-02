package model

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// profileOrdinalsFile pins each enum's constants, in the order that numbers them.
const profileOrdinalsFile = "testdata/profile_ordinals.json"

// TestEveryProfileOrdinalKeepsItsName: architecture_profiles.json holds raw numbers, so a reorder of an enum gives old
// numbers new meanings with every identity unchanged, and old evidence still reads as valid. Every enum the profile reaches,
// through its nested policies too, must name its constants in the pinned order. An addition at the end is an edit of the
// pin; anything else in it republishes every model's definition, and the old numbers must be migrated first.
func TestEveryProfileOrdinalKeepsItsName(t *testing.T) {
	t.Parallel()
	declared := declaredEnums(t)
	reached := map[string][]string{}
	for name := range profileEnums(reflect.TypeFor[ArchitectureProfile](), map[reflect.Type]bool{}) {
		if constants, enum := declared[name]; enum {
			reached[name] = constants
		}
	}
	raw, err := os.ReadFile(profileOrdinalsFile)
	if err != nil {
		t.Fatal(err)
	}
	var pinned map[string][]string
	if err := json.Unmarshal(raw, &pinned); err != nil {
		t.Fatal(err)
	}
	for name, constants := range reached {
		if !slices.Equal(pinned[name], constants) {
			t.Errorf("%s numbers %v, and %s pins %v", name, constants, profileOrdinalsFile, pinned[name])
		}
	}
	for name := range pinned {
		if _, held := reached[name]; !held {
			t.Errorf("%s pins %s, which the profile no longer reaches", profileOrdinalsFile, name)
		}
	}
	if len(reached) == 0 {
		t.Fatal("the walk reached no enum")
	}
}

// profileEnums names the integer types of this package a type reaches, through fields, elements, keys and pointers.
func profileEnums(kind reflect.Type, seen map[reflect.Type]bool) map[string]bool {
	found := map[string]bool{}
	if seen[kind] {
		return found
	}
	seen[kind] = true
	switch kind.Kind() {
	case reflect.Struct:
		for field := range kind.Fields() {
			for name := range profileEnums(field.Type, seen) {
				found[name] = true
			}
		}
	case reflect.Map:
		for name := range profileEnums(kind.Key(), seen) {
			found[name] = true
		}
		fallthrough
	case reflect.Pointer, reflect.Slice, reflect.Array:
		for name := range profileEnums(kind.Elem(), seen) {
			found[name] = true
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if kind.PkgPath() == reflect.TypeFor[ArchitectureProfile]().PkgPath() {
			found[kind.Name()] = true
		}
	}
	return found
}

// declaredEnums reads each named type's constants from this package's source in declaration order, a blank one included,
// since the blank holds a number. A constant with no type of its own takes the one before it in its block.
func declaredEnums(t *testing.T) map[string][]string {
	t.Helper()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(files)
	enums := map[string][]string{}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range parsed.Decls {
			block, constants := declaration.(*ast.GenDecl)
			if !constants || block.Tok != token.CONST {
				continue
			}
			named := ""
			for _, spec := range block.Specs {
				value := spec.(*ast.ValueSpec)
				if value.Type != nil {
					named = ""
					if identifier, simple := value.Type.(*ast.Ident); simple {
						named = identifier.Name
					}
				}
				if named == "" {
					continue
				}
				for _, name := range value.Names {
					enums[named] = append(enums[named], name.Name)
				}
			}
		}
	}
	return enums
}
