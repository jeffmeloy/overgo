package runrecord

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"

	"overgo/internal/artifact"
)

// uniquePublicationKey prevents a compare-and-set contender from being
// mistaken for an idempotent replay of another contender's batch.
func uniquePublicationKey(prefix string, id artifact.ID) (string, error) {
	var nonce [sha256.Size]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s%s/%x", prefix, id, nonce[:]), nil
}
