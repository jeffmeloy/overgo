package processmeasure

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

const helperEnvironment = "OVERGO_PROCESS_MEASURE_HELPER"

func TestMeasureTracksChildPeakWorkingSet(t *testing.T) {
	if os.Getenv(helperEnvironment) == "1" {
		memory := make([]byte, 32<<20)
		for index := 0; index < len(memory); index += 4096 {
			memory[index] = byte(index)
		}
		fmt.Printf("resident=%d\n", len(memory))
		time.Sleep(30 * time.Millisecond)
		runtime.KeepAlive(memory)
		return
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestMeasureTracksChildPeakWorkingSet$")
	command.Env = append(os.Environ(), helperEnvironment+"=1")
	result, err := Measure(command)
	if err != nil {
		t.Fatalf("child: %v: %s", err, result.Output)
	}
	if !strings.Contains(string(result.Output), "resident=33554432") {
		t.Fatalf("child output = %q", result.Output)
	}
	if result.PeakWorkingSetByte < 32<<20 || result.Wall <= 0 {
		t.Fatalf("measurement wall=%s peak=%d", result.Wall, result.PeakWorkingSetByte)
	}
	t.Logf("child wall=%s peak=%.3f MiB", result.Wall, float64(result.PeakWorkingSetByte)/(1<<20))
}
