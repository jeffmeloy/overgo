package quant

import (
	"strings"

	"overgo/internal/tensor/dtype"
)

type codecPolicy struct {
	dataType   dtype.Type
	importance bool
	supported  bool
}

var codecsByType = func() [dtype.Count]codecPolicy {
	var result [dtype.Count]codecPolicy
	for _, value := range codecCatalog {
		value.supported = true
		result[value.dataType] = value
	}
	return result
}()

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
	for _, codec := range codecCatalog {
		if normalized == strings.ToLower(codec.dataType.String()) {
			return codec.dataType, true
		}
	}
	return 0, false
}

// CanQuantize: supported destination check.
func CanQuantize(dataType dtype.Type) bool {
	return dataType < dtype.Count && codecsByType[dataType].supported
}

// RequiresImportance: destination requires explicit element weights.
func RequiresImportance(dataType dtype.Type) bool {
	return dataType < dtype.Count && codecsByType[dataType].importance
}
