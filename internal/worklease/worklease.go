// Package worklease validates the Git commits, plan identifiers and bounded text that automation records carry.
//   - a package that checks these identifiers never compiles the plan's readers, because this package sits below the plan
package worklease

import "strings"

// ValidCommit accepts a lower-case SHA-1 or SHA-256 Git identity.
func ValidCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	for _, char := range value {
		if char < '0' || char > '9' && char < 'a' || char > 'f' {
			return false
		}
	}
	return true
}

// NonCanonicalPathSeparator is the separator persisted worktree paths never carry; they use forward slashes on every host.
const NonCanonicalPathSeparator = "\\"
