package densecausal

import (
	"overgo/internal/testskip"
	"testing"
)

func requireLongTest(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
}
