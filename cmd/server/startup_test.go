package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
)

func TestDeviceStartupErrorTypesAllocationFailure(t *testing.T) {
	// CUDA_ERROR_OUT_OF_MEMORY is result 2; the text says nothing about memory.
	allocation := fmt.Errorf("load model: %w", &driver.ResultError{Code: 2, Message: "controlled allocation diagnostic"})
	if typed := deviceStartupError(allocation); !errors.Is(typed, processcontrol.ErrDeviceMemory) || !strings.Contains(typed.Error(), allocation.Error()) {
		t.Fatalf("allocation failure typed as %v", typed)
	}
	if typed := deviceStartupError(nil); typed != nil {
		t.Fatalf("success retyped as %v", typed)
	}
	for _, other := range []error{errors.New("out of memory text without a driver result"), fmt.Errorf("load device: %w", processcontrol.ErrResourceBusy)} {
		if typed := deviceStartupError(other); errors.Is(typed, processcontrol.ErrDeviceMemory) || typed.Error() != other.Error() {
			t.Fatalf("%v retyped as %v", other, typed)
		}
	}
}
