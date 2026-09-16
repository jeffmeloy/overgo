package agenttool

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/processcontrol"
)

func TestArgvOutputLimit(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", filepath.Dir(executable)+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, mode := range []string{"finite", "continuing-tree", "valid"} {
		t.Run(mode, func(t *testing.T) {
			marker := filepath.Join(t.TempDir(), "heartbeat")
			manual := inspectionManual(t, "output.fixture", Transport{
				Kind: TransportArgv, Program: filepath.Base(executable),
				Args: []string{"-test.run=^TestArgvOutputProcess$", "--", mode, marker},
			})
			ctx := t.Context()
			result, effect, err := NewExecutor().InvokeWithEffect(ctx, manual, json.RawMessage(`{"pattern":"x"}`))
			if mode == "valid" {
				var actual string
				if err != nil || json.Unmarshal(result, &actual) != nil || actual != "hello 世界\t\"quoted\"" {
					t.Fatalf("valid stdout: result=%s err=%v", result, err)
				}
				return
			}
			limit, limited := errors.AsType[*OutputLimitError](err)
			if !limited || ctx.Err() != nil {
				t.Fatalf("overflow must fail before the fixture deadline: err=%v context=%v result bytes=%d", err, ctx.Err(), len(result))
			}
			if limit.RetainedBytes != artifact.MaxContentBytes || limit.Receipt.StdoutBytes <= artifact.MaxContentBytes || limit.Receipt.WallNS <= 0 {
				t.Fatalf("overflow receipt does not bind the measured bound: %+v", limit)
			}
			if len(result) != 0 || !reflect.DeepEqual(effect, InvocationEffect{}) {
				t.Fatalf("overflow returned successful result/effect: %d %+v", len(result), effect)
			}
			if mode == "continuing-tree" {
				if !limit.Receipt.TreeTerminated {
					t.Fatalf("continuing tree not terminated: %+v", limit.Receipt)
				}
				// The fixture recorded its descendant's pid before overflowing;
				// the terminated tree leaves that process dead.
				recorded, err := os.ReadFile(marker)
				if err != nil {
					t.Fatalf("descendant did not start: %v", err)
				}
				descendant, err := strconv.Atoi(strings.TrimSpace(string(recorded)))
				if err != nil || descendant <= 0 {
					t.Fatalf("descendant pid %q: %v", recorded, err)
				}
				if processcontrol.ProcessAlive(descendant) {
					t.Fatalf("descendant %d survived overflow", descendant)
				}
			}
		})
	}
	t.Run("retained capacity", func(t *testing.T) {
		cancelled := false
		output := argvOutput{cancel: func() { cancelled = true }}
		chunk := bytes.Repeat([]byte("x"), 100003)
		for len(output.data) < artifact.MaxContentBytes {
			amount := min(len(chunk), artifact.MaxContentBytes-len(output.data))
			if n, err := output.Write(chunk[:amount]); err != nil || n != amount {
				t.Fatalf("write = %d, %v", n, err)
			}
			if cap(output.data) > artifact.MaxContentBytes || cancelled {
				t.Fatalf("premature cancellation or excessive backing capacity: %d", cap(output.data))
			}
		}
		if n, err := output.Write(chunk); err != nil || n != len(chunk) || !cancelled || !output.overflow || len(output.data) != artifact.MaxContentBytes || cap(output.data) > artifact.MaxContentBytes {
			t.Fatalf("overflow collector: n=%d err=%v cancelled=%t retained=%d capacity=%d", n, err, cancelled, len(output.data), cap(output.data))
		}
	})
}

// TestArgvOutputProcess is the hermetic child executable; normal test runs do
// nothing. The flood is bounded even against the unfixed adapter.
func TestArgvOutputProcess(t *testing.T) {
	separator := -1
	for index, arg := range os.Args {
		if arg == "--" {
			separator = index
			break
		}
	}
	if separator < 0 {
		return
	}
	args := os.Args[separator+1:]
	if len(args) != 2 {
		os.Exit(2)
	}
	mode, marker := args[0], args[1]
	if mode == "valid" {
		_, _ = io.WriteString(os.Stdout, "hello 世界\t\"quoted\"\r\n")
		os.Exit(0)
	}
	if mode == "descendant" {
		// The descendant announces itself, then lives until the tree ends.
		fmt.Println("descendant ready")
		select {}
	}
	if mode == "continuing-tree" {
		executable, err := os.Executable()
		if err != nil {
			os.Exit(2)
		}
		child := exec.Command(executable, "-test.run=^TestArgvOutputProcess$", "--", "descendant", marker)
		child.Stderr = os.Stderr
		ready, err := child.StdoutPipe()
		if err != nil || child.Start() != nil {
			os.Exit(2)
		}
		if _, err := bufio.NewReader(ready).ReadString('\n'); err != nil {
			os.Exit(2)
		}
		// The parent test reads the descendant's pid from the marker.
		if err := os.WriteFile(marker, []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			os.Exit(2)
		}
	}
	chunk := bytes.Repeat([]byte("x"), 64<<10)
	for written := 0; written <= artifact.MaxContentBytes; {
		count, err := os.Stdout.Write(chunk[:min(len(chunk), artifact.MaxContentBytes+1-written)])
		if err != nil {
			os.Exit(2)
		}
		written += count
	}
	if mode == "continuing-tree" {
		// Output continues until the tree is terminated; a stopped reader
		// blocks the write itself.
		for {
			if _, err := os.Stdout.Write(chunk); err != nil {
				os.Exit(2)
			}
		}
	}
	os.Exit(0)
}
