package strictjson

import "strconv"

// ParseInteger parses the signed decimal integer domain accepted by JSON
// schema integer values.
func ParseInteger(text string) (int64, error) {
	return strconv.ParseInt(text, 10, 64)
}

// ParseNumber parses the binary64 number domain accepted by JSON schema.
func ParseNumber(text string) (float64, error) {
	return strconv.ParseFloat(text, 64)
}
