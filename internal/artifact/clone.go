package artifact

import "slices"

func IDPointer(id ID) *ID { return &id }

// ClonePointer returns an independent copy of a pointed-to value; nil stays
// nil. This is the single owner of the optional-field copy idiom.
func ClonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func CloneID(id *ID) *ID { return ClonePointer(id) }

func CloneAliasBindings(bindings []AliasBinding) []AliasBinding {
	result := slices.Clone(bindings)
	for index := range result {
		result[index].Previous = CloneID(result[index].Previous)
	}
	return result
}
