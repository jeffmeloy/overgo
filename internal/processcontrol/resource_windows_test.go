//go:build windows

package processcontrol

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestResourceContention(t *testing.T) {
	if name := os.Getenv("OVERGO_TEST_RESOURCE_CLAIM"); name != "" {
		err := ClaimResource(name)
		if os.Getenv("OVERGO_TEST_RESOURCE_OWNER") == "1" {
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("reserved")
			_, _ = io.Copy(io.Discard, os.Stdin)
		} else if !errors.Is(err, ErrResourceBusy) {
			t.Fatalf("contention lost its type: %v", err)
		}
		return
	}
	ctx := t.Context()
	environment := append(os.Environ(), "OVERGO_TEST_RESOURCE_CLAIM="+t.TempDir())
	owner := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestResourceContention$")
	owner.Env = append(environment, "OVERGO_TEST_RESOURCE_OWNER=1")
	input, err := owner.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := owner.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	owner.Stderr = os.Stderr
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { input.Close(); owner.Process.Kill(); owner.Wait() }()
	line, err := bufio.NewReader(output).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "reserved" {
		t.Fatalf("resource owner: %q %v", line, err)
	}
	contender := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestResourceContention$")
	contender.Env = environment
	if output, err := contender.CombinedOutput(); err != nil {
		t.Fatalf("independent contender: %s %v", output, err)
	}
}
