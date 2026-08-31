package gitauthority

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
)

// ValidObjectID reports whether value is a hexadecimal SHA-1 or SHA-256 Git
// object identifier.
func ValidObjectID(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && (len(decoded) == sha1.Size || len(decoded) == sha256.Size)
}
