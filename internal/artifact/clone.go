package artifact

import (
	"fmt"
	"reflect"
	"slices"
)

func IDPointer(id ID) *ID { return new(id) }

// ClonePointer returns an independent copy of a pointed-to value; nil stays
// nil. This is the single owner of the optional-field copy idiom.
func ClonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	return new(*value)
}

func CloneID(id *ID) *ID { return ClonePointer(id) }

func CloneAliasBindings(bindings []AliasBinding) []AliasBinding {
	result := slices.Clone(bindings)
	for index := range result {
		result[index].Previous = CloneID(result[index].Previous)
	}
	return result
}

// DeepCopy returns the copy a document codec makes of a T before it canonicalizes or hands the value out.
//   - the copy shares no slice, map or pointer target with its original
//   - a nil slice, map or pointer stays nil and an empty one stays empty, because a document's JSON identity tells null from []
//   - an interface, func or channel value and a map key are copied as they are
//   - DeepCopy panics when T holds an unexported field other than an embedded struct with a slice, map or pointer, because reflection cannot replace it
//   - so a codec of such a type is refused when its package initializes and never while it copies a document.
func DeepCopy[T any]() func(T) T {
	if err := copyRefusal(reflect.TypeFor[T](), map[reflect.Type]bool{}); err != nil {
		panic(fmt.Sprintf("artifact: no deep copy of %s: %v", reflect.TypeFor[T](), err))
	}
	return func(value T) T {
		copyReferences(reflect.ValueOf(&value).Elem(), reflect.ValueOf(value))
		return value
	}
}

// copyReferences replaces each slice, map and pointer target that dst shares with src by a copy.
//   - dst equals src on entry
func copyReferences(dst, src reflect.Value) {
	switch src.Kind() {
	case reflect.Pointer:
		if !src.IsNil() {
			dst.Set(reflect.New(src.Type().Elem()))
			dst.Elem().Set(src.Elem())
			copyReferences(dst.Elem(), src.Elem())
		}
	case reflect.Map:
		if !src.IsNil() {
			dst.Set(reflect.MakeMapWithSize(src.Type(), src.Len()))
			for entry := src.MapRange(); entry.Next(); {
				value := reflect.New(src.Type().Elem()).Elem()
				value.Set(entry.Value())
				copyReferences(value, entry.Value())
				dst.SetMapIndex(entry.Key(), value)
			}
		}
	case reflect.Struct:
		for index := range src.NumField() {
			if field := src.Type().Field(index); copied(field) && holdsReference(field.Type) {
				copyReferences(dst.Field(index), src.Field(index))
			}
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Len()))
		reflect.Copy(dst, src)
		fallthrough
	case reflect.Array:
		if holdsReference(src.Type().Elem()) {
			for index := range src.Len() {
				copyReferences(dst.Index(index), src.Index(index))
			}
		}
	}
}

// copied reports whether copyReferences reaches a struct field.
//   - reflection sets an exported field and the exported fields of an embedded struct, and no other unexported field
func copied(field reflect.StructField) bool {
	return field.IsExported() || field.Anonymous && field.Type.Kind() == reflect.Struct
}

// holdsReference reports whether assigning a value of t shares a slice, map or pointer target.
func holdsReference(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Map:
		return true
	case reflect.Array:
		return holdsReference(t.Elem())
	case reflect.Struct:
		for field := range t.Fields() {
			if holdsReference(field.Type) {
				return true
			}
		}
	}
	return false
}

// copyRefusal names the first field within t that holds a slice, map or pointer and that copyReferences cannot reach.
//   - nil <=> a copy of any t value shares no slice, map or pointer target with it
func copyRefusal(t reflect.Type, seen map[reflect.Type]bool) error {
	if seen[t] {
		return nil
	}
	seen[t] = true
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map:
		return copyRefusal(t.Elem(), seen)
	case reflect.Struct:
		for field := range t.Fields() {
			if !copied(field) && holdsReference(field.Type) {
				return fmt.Errorf("unexported field %s.%s of type %s holds a reference the copy would share", t, field.Name, field.Type)
			}
			if err := copyRefusal(field.Type, seen); err != nil {
				return err
			}
		}
	}
	return nil
}
