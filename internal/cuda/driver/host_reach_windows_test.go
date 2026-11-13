package driver

import (
	"strings"
	"testing"
)

// TestATestWithoutTheCUDASwitchIsRefusedTheCard holds the card to the CUDA lane's tests (card-reach): a test run without
// OVERGO_CUDA_TEST=1 asking for a context is refused before the card's turn is taken, naming the switch and the gate,
// where on 2026-11-13 the trainer tests of three packages each waited about a minute for a render's picture in a
// landing's host stages. With the switch set the refusal is not the guard's: here the library holds no driver, and it
// says so.
func TestATestWithoutTheCUDASwitchIsRefusedTheCard(t *testing.T) {
	t.Setenv("OVERGO_CUDA_TEST", "")
	_, err := (&Library{}).ContextCreate(0, 0)
	if err == nil || !strings.Contains(err.Error(), "OVERGO_CUDA_TEST") || !strings.Contains(err.Error(), "cudatest.Require") {
		t.Fatalf("a test without the CUDA switch asking for a context = %v, want it refused naming the switch and the gate", err)
	}
	t.Setenv("OVERGO_CUDA_TEST", "1")
	if _, err := (&Library{}).ContextCreate(0, 0); err == nil || strings.Contains(err.Error(), "cudatest.Require") {
		t.Fatalf("a test with the CUDA switch asking for a context = %v, want the library's own refusal, not the guard's", err)
	}
}
