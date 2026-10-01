package main

import (
	"cmp"
	"os"
)

// retainedReferenceStore shares the smoke lane's explicit read-only evidence
// source. It never redirects the general data root or writable test fixtures.
func retainedReferenceStore(local string) string {
	return cmp.Or(os.Getenv("OVERGO_SMOKE_REFERENCE_STORE"), local)
}
