package artifact

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// A probe reference is absent, empty or populated.
const (
	probeAbsent = iota
	probeEmpty
	probePopulated
)

const (
	// probeLength is the element count of a populated probe slice or map, 2 so that a copy that keeps one element differs.
	probeLength = 2
	// probeDepth is the most slices, maps and pointers above a filled one, 2 so that a child's child's lists are filled.
	probeDepth = 2
)

// probeState is one fill variant.
//   - a leaf is a slice, map or pointer whose elements hold no reference, and a branch is any other one
type probeState struct{ leaf, branch int }

// probeStates put every reference in each state while the references above it are populated.
var probeStates = []probeState{
	{probePopulated, probePopulated},
	{probeEmpty, probePopulated},
	{probeAbsent, probePopulated},
	{probePopulated, probeEmpty},
	{probePopulated, probeAbsent},
}

// keptClones names each codec whose copy stays hand-written, with the reason DeepCopy cannot replace it.
var keptClones = map[string]string{
	"internal/adaptiveparity/scratch_oracle.go":  "the copy makes a nil token list empty, and identity tells null from []",
	"internal/composition/execution_plan.go":     "the copy makes empty bridge, operator and session lists nil, and identity tells null from []",
	"internal/controllertrain/promotion.go":      "the copy trims each challenger's refusal",
	"internal/modelartifact/tensor_inventory.go": "the copy makes a nil tensor list empty, and identity tells null from []",
	"internal/modelrecipe/model_definition.go":   "model.Spec holds an unexported profile pointer, which DeepCopy refuses",
	"internal/tokenizer/vocabulary_artifact.go":  "the copy makes a nil token list empty, and identity tells null from []",
}

type cloneProbe struct {
	Names    []string
	Raw      json.RawMessage
	Rows     []cloneProbeRow
	Optional *cloneProbeRow
	Index    map[string][]int
	Fixed    [probeLength]cloneProbeRow
	Children []cloneProbe
	Any      any
	Hook     func() int
	cloneProbeTags
	note string
}

type cloneProbeRow struct {
	Label  string
	Values []float32
	Next   *int
	Owner  ID
}

type cloneProbeTags struct {
	Tags []string
}

type cloneRefused struct {
	Name  string
	cache []int
}

type probeFiller struct {
	states probeState
	next   int
}

// fill sets every value reachable from value through exported and embedded fields to a distinct non-zero value.
//   - depth counts the slices, maps and pointers above value, and one deeper than probeDepth stays absent
func (f *probeFiller) fill(value reflect.Value, depth int) {
	kind := value.Kind()
	state, length := f.states.branch, 0
	if kind == reflect.Pointer || kind == reflect.Slice || kind == reflect.Map {
		if !holdsReference(value.Type().Elem()) {
			state = f.states.leaf
		}
		if depth > probeDepth {
			state = probeAbsent
		}
	}
	if state == probePopulated {
		length = probeLength
	}
	f.next++
	switch {
	case value.CanInt():
		value.SetInt(int64(f.next))
	case value.CanUint():
		value.SetUint(uint64(f.next))
	case value.CanFloat():
		value.SetFloat(float64(f.next))
	case kind == reflect.Bool:
		value.SetBool(true)
	case kind == reflect.String:
		value.SetString("v" + strconv.Itoa(f.next))
	case kind == reflect.Interface:
		value.Set(reflect.ValueOf("any" + strconv.Itoa(f.next)))
	case kind == reflect.Pointer && state != probeAbsent:
		value.Set(reflect.New(value.Type().Elem()))
		f.fill(value.Elem(), depth+1)
	case kind == reflect.Slice && state != probeAbsent:
		value.Set(reflect.MakeSlice(value.Type(), length, length))
		for index := range length {
			f.fill(value.Index(index), depth+1)
		}
	case kind == reflect.Map && state != probeAbsent:
		value.Set(reflect.MakeMap(value.Type()))
		for range length {
			key, element := reflect.New(value.Type().Key()).Elem(), reflect.New(value.Type().Elem()).Elem()
			f.fill(key, depth+1)
			f.fill(element, depth+1)
			value.SetMapIndex(key, element)
		}
	case kind == reflect.Array:
		for index := range value.Len() {
			f.fill(value.Index(index), depth)
		}
	case kind == reflect.Struct:
		for index := range value.NumField() {
			if copied(value.Type().Field(index)) {
				f.fill(value.Field(index), depth)
			}
		}
	}
}

// changeEach changes every value reachable from value in place, so that a reference shared with another value changes that value too.
func changeEach(value reflect.Value) {
	switch kind := value.Kind(); {
	case value.CanInt():
		value.SetInt(value.Int() + 1)
	case value.CanUint():
		value.SetUint(value.Uint() + 1)
	case value.CanFloat():
		value.SetFloat(value.Float() + 1)
	case kind == reflect.String:
		value.SetString(value.String() + "changed")
	case kind == reflect.Pointer && !value.IsNil():
		changeEach(value.Elem())
	case kind == reflect.Map:
		for entry := value.MapRange(); entry.Next(); {
			element := reflect.New(value.Type().Elem()).Elem()
			element.Set(entry.Value())
			changeEach(element)
			value.SetMapIndex(entry.Key(), element)
		}
	case kind == reflect.Slice || kind == reflect.Array:
		for index := range value.Len() {
			changeEach(value.Index(index))
		}
	case kind == reflect.Struct:
		for index := range value.NumField() {
			if copied(value.Type().Field(index)) {
				changeEach(value.Field(index))
			}
		}
	}
}

func filledProbe(states probeState) cloneProbe {
	var probe cloneProbe
	(&probeFiller{states: states}).fill(reflect.ValueOf(&probe).Elem(), 0)
	probe.note = "unexported"
	return probe
}

// TestDefaultCloneMatchesEveryReplacedClone holds DeepCopy to what every replaced hand-written clone returned.
//   - the copy equals its original, and reflect.DeepEqual tells a nil slice or map from an empty one
//   - changing every value the copy reaches leaves its original unchanged
//   - interface and func values are copied as they are
//   - a codec of a type with an unexported reference is refused when it is constructed
//   - every hand-written clone left is named in keptClones with its reason
func TestDefaultCloneMatchesEveryReplacedClone(t *testing.T) {
	t.Parallel()
	copyProbe := DeepCopy[cloneProbe]()
	for _, states := range probeStates {
		original := filledProbe(states)
		copied := copyProbe(original)
		if !reflect.DeepEqual(copied, original) {
			t.Fatalf("states %v: copy differs from its original:\n%#v\n%#v", states, copied, original)
		}
		changeEach(reflect.ValueOf(&copied).Elem())
		if !reflect.DeepEqual(original, filledProbe(states)) {
			t.Fatalf("states %v: changing the copy changed its original", states)
		}
	}

	shared := new(int)
	original := cloneProbe{Any: shared, Hook: func() int { return 1 }}
	copied := copyProbe(original)
	if copied.Any != any(shared) || copied.Hook == nil || copied.Hook() != original.Hook() {
		t.Fatalf("copy replaced an interface or func value: %#v", copied)
	}

	refused := func() (message any) {
		defer func() { message = recover() }()
		JSONDocumentCodec[cloneRefused]("refused", KindEvidence, "application/json", "overgo/refused/v1", nil, nil, nil)
		return nil
	}()
	if refused == nil || !strings.Contains(refused.(string), "cache") {
		t.Fatalf("codec of a type with an unexported slice was constructed, panic = %v", refused)
	}

	found := handWrittenClones(t)
	for file, reason := range keptClones {
		if !found[file] {
			t.Errorf("%s keeps no hand-written clone, so its entry (%s) is stale", file, reason)
		}
	}
	for file := range found {
		if keptClones[file] == "" {
			t.Errorf("%s holds a hand-written clone that keptClones does not name; DeepCopy is the codec's copy", file)
		}
	}
}

// handWrittenClones returns the production files whose document codec copies by other than DeepCopy.
func handWrittenClones(t *testing.T) map[string]bool {
	t.Helper()
	root := filepath.Join("..", "..")
	found := map[string]bool{}
	for _, tree := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, tree), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			source, err := os.ReadFile(path)
			if err != nil || !bytes.Contains(source, []byte("Clone")) {
				return err
			}
			file, err := parser.ParseFile(token.NewFileSet(), path, source, 0)
			if err != nil {
				return err
			}
			ast.Inspect(file, func(node ast.Node) bool {
				var clone ast.Expr
				switch node := node.(type) {
				case *ast.CallExpr:
					if method, ok := node.Fun.(*ast.SelectorExpr); ok && method.Sel.Name == "WithClone" && len(node.Args) == 1 {
						clone = node.Args[0]
					}
				case *ast.KeyValueExpr:
					if key, ok := node.Key.(*ast.Ident); ok && key.Name == "Clone" {
						clone = node.Value
					}
				}
				if call, ok := clone.(*ast.CallExpr); ok {
					if index, ok := call.Fun.(*ast.IndexExpr); ok && strings.TrimPrefix(types.ExprString(index.X), "artifact.") == "DeepCopy" {
						clone = nil
					}
				}
				if clone != nil {
					relative, _ := filepath.Rel(root, path)
					found[filepath.ToSlash(relative)] = true
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return found
}
