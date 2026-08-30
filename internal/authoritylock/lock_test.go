package authoritylock

import "testing"

func TestAcquireSerializesPlanAndGateMutation(t *testing.T) {
	repository := t.TempDir()
	first, err := Acquire(repository)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := Acquire(repository); err == nil {
		_ = second.Close()
		t.Fatal("concurrent authority mutation acquired the shared lock")
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Acquire(repository)
	if err != nil {
		t.Fatalf("reacquire authority lock: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}
