// Package textcheck owns bounded canonical-text predicates shared by document schemas.
package textcheck

import "strings"

func Bounded(value string, maxBytes int, forbidden string) bool {
	return value != "" && maxBytes > 0 && len(value) <= maxBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, forbidden)
}
