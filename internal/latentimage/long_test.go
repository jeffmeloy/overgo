package latentimage

import (
	"overgo/internal/testskip"
	"testing"
)

func requireLongTest(t *testing.T) {
	t.Helper()
	testskip.Short(t)
}
