// Package textcheck owns bounded canonical-text predicates shared by document schemas.
package textcheck

import "strings"

func Bounded(value string, maxBytes int, forbidden string) bool {
	return value != "" && maxBytes > 0 && len(value) <= maxBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, forbidden)
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
