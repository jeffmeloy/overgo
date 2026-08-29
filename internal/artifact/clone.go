package artifact

import "slices"

func IDPointer(id ID) *ID { return new(id) }

func CloneID(id *ID) *ID {
	if id == nil {
		return nil
	}
	return IDPointer(*id)
}

func CloneAliasBindings(bindings []AliasBinding) []AliasBinding {
	result := slices.Clone(bindings)
	for index := range result {
		result[index].Previous = CloneID(result[index].Previous)
	}
	return result
}
