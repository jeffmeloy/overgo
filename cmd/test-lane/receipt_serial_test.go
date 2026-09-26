package main

import (
	"slices"
	"testing"
)

// TestReceiptRunsPackagesOneAtATime holds a receipt to running its packages
// serially without the operator asking -- parallel device packages reserve
// the one GPU against each other -- while an operator's own -p still wins.
func TestReceiptRunsPackagesOneAtATime(t *testing.T) {
	t.Parallel()
	args := receiptArgs([]string{"-tags", "integration", "./internal/cuda/driver"})
	if !slices.Equal(args, []string{"test", "-json", "-count=1", "-p=1", "-tags", "integration", "./internal/cuda/driver"}) {
		t.Fatalf("receipt go test args = %v", args)
	}
	if args := receiptArgs([]string{"-p", "4", "./x"}); slices.Index(args, "-p=1") > slices.Index(args, "-p") {
		t.Fatalf("an operator -p does not follow the default: %v", args)
	}
}
