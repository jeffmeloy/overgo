package artifact

import (
	"encoding/json"
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

}
