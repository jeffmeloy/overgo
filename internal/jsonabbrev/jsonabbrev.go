// Package jsonabbrev owns bounded rendering of JSON-shaped values:
// prompts and settings stay readable while bulk fields -- embedded
// tensors, payload arrays, oversized strings -- state their extent
// instead of their contents. It is the single abbreviation authority
// for request digests and report rendering.
package jsonabbrev

import "fmt"

// Bounds: a field longer than these is bulk data, not a setting a
// reader compares.
const (
	// StringRunes bounds a rendered string; longer strings elide with
	// an ellipsis so prompts stay readable without flooding output.
	StringRunes = 200
	// ArrayItems bounds a rendered array; longer arrays state their
	// extent instead of their values.
	ArrayItems = 8
)

// Value abbreviates one decoded JSON value recursively.
func Value(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, entry := range typed {
			result[key] = Value(entry)
		}
		return result
	case []any:
		if len(typed) > ArrayItems {
			return fmt.Sprintf("[%d values]", len(typed))
		}
		result := make([]any, len(typed))
		for index, entry := range typed {
			result[index] = Value(entry)
		}
		return result
	case string:
		if len(typed) > StringRunes {
			return typed[:StringRunes] + "…"
		}
		return typed
	}
	return value
}
