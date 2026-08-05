package quant

import (
	"strings"

	"llamacpp2go/internal/tensor/dtype"
)

type codecPolicy struct {
	dataType   dtype.Type
	importance bool
}

var codecCatalog = []codecPolicy{
	{dataType: dtype.F32},
	{dataType: dtype.F16},
	{dataType: dtype.BF16},
	{dataType: dtype.Q1_0},
	{dataType: dtype.Q2_0},
	{dataType: dtype.Q4_0},
	{dataType: dtype.Q4_1},
	{dataType: dtype.Q5_0},
	{dataType: dtype.Q5_1},
	{dataType: dtype.Q8_0},
	{dataType: dtype.Q8_1},
	{dataType: dtype.Q2K},
	{dataType: dtype.Q3K},
	{dataType: dtype.Q4K},
	{dataType: dtype.Q5K},
	{dataType: dtype.Q6K},
	{dataType: dtype.Q8K},
	{dataType: dtype.TQ1_0},
	{dataType: dtype.TQ2_0},
	{dataType: dtype.MXFP4},
	{dataType: dtype.NVFP4},
	{dataType: dtype.IQ4NL},
	{dataType: dtype.IQ4XS},
	{dataType: dtype.IQ2S},
	{dataType: dtype.IQ2XXS, importance: true},
	{dataType: dtype.IQ2XS, importance: true},
	{dataType: dtype.IQ1S, importance: true},
	{dataType: dtype.IQ1M, importance: true},
	{dataType: dtype.IQ3XXS},
	{dataType: dtype.IQ3S},
}

var codecByType = func() map[dtype.Type]codecPolicy {
	result := make(map[dtype.Type]codecPolicy, len(codecCatalog))
	for _, codec := range codecCatalog {
		result[codec.dataType] = codec
	}
	return result
}()

var codecByName = func() map[string]dtype.Type {
	result := make(map[string]dtype.Type, len(codecCatalog))
	for _, codec := range codecCatalog {
		result[strings.ToLower(codec.dataType.String())] = codec.dataType
	}
	return result
}()

// Types: supported quantization destinations.
func Types() []dtype.Type {
	result := make([]dtype.Type, len(codecCatalog))
	for index, codec := range codecCatalog {
		result[index] = codec.dataType
	}
	return result
}

// TypeNames: supported destination names.
func TypeNames() []string {
	result := make([]string, len(codecCatalog))
	for index, codec := range codecCatalog {
		result[index] = strings.ToLower(codec.dataType.String())
	}
	return result
}

// ParseType: normalized supported destination lookup.
func ParseType(value string) (dtype.Type, bool) {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	dataType, ok := codecByName[normalized]
	return dataType, ok
}

// CanQuantize: supported destination check.
func CanQuantize(dataType dtype.Type) bool {
	_, ok := codecByType[dataType]
	return ok
}

// RequiresImportance: destination requires explicit element weights.
func RequiresImportance(dataType dtype.Type) bool {
	return codecByType[dataType].importance
}
