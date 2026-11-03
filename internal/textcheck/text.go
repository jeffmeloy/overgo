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
		if !LowerIdentifierByte(value[index]) {
			return false
		}
	}
	return true
}

// LowerIdentifierByte reports whether one byte belongs to the lower-case
// identifier alphabet: a lower-case letter, a digit, '.', '-', or '_'.
func LowerIdentifierByte(character byte) bool {
	return character >= 'a' && character <= 'z' || character >= '0' && character <= '9' ||
		character == '.' || character == '-' || character == '_'
}

// The character sets Only reads names against.
const (
	// Digits are the decimal digits.
	Digits = "0123456789"
	// HexDigits are the hexadecimal digits, in both cases.
	HexDigits = Digits + "abcdefABCDEF"
	// LowerAlphanumeric are the lower-case letters and the digits.
	LowerAlphanumeric = "abcdefghijklmnopqrstuvwxyz" + Digits
	// Alphanumeric are the letters, in both cases, and the digits.
	Alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + LowerAlphanumeric
)

// Only says value is not empty and holds the given characters alone.
func Only(value, characters string) bool {
	return value != "" && strings.Trim(value, characters) == ""
}
