package evaluation

import (
	"cmp"
	"os"
)

// retainedReferenceStore uses the smoke lane's explicit read-only fixture
// source while leaving general data roots and writable test stores local.
func retainedReferenceStore(local string) string {
	return cmp.Or(os.Getenv("OVERGO_SMOKE_REFERENCE_STORE"), local)
}
