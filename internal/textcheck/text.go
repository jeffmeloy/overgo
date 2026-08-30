// Package textcheck owns bounded canonical-text predicates shared by document schemas.
package textcheck

import (
	"strings"
	"unicode"
)

func Bounded(value string, maxBytes int, forbidden string) bool {
	return value != "" && maxBytes > 0 && len(value) <= maxBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, forbidden)
}

// BoundedToken accepts one canonical, bounded token with no Unicode whitespace,
// control characters, or caller-owned separators.
func BoundedToken(value string, maxBytes int, forbidden string) bool {
	return Bounded(value, maxBytes, forbidden) && strings.IndexFunc(value, func(character rune) bool {
		return unicode.IsSpace(character) || unicode.IsControl(character)
	}) < 0
}

func LowerIdentifier(value string, maxBytes int) bool {
	if !Bounded(value, maxBytes, "") {
		return false
	}
	for index := range len(value) {
		character := value[index]
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') &&
			character != '.' && character != '-' && character != '_' {
			return false
		}
	}
	return true
}
