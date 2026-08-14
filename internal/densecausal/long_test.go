package densecausal

import (
	"testing"

	"overgo/internal/testevidence"
)

func requireLongTest(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
}
