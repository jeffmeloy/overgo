package hfrepo

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSourceReceipt(t *testing.T) {
	makeSource := func(t *testing.T) (string, SourceReceipt) {
		t.Helper()
		root := t.TempDir()
		receipt := SourceReceipt{Source: "https://www.modelscope.cn/openmoss/tiny.git", Revision: "797fa8e678b4f0a766271bf5103e6cd7a2faa999"}
		for _, name := range []string{"weights.bin", "config.json"} {
			data := []byte(name)
			if err := os.WriteFile(filepath.Join(root, name), data, 0600); err != nil {
				t.Fatal(err)
			}
			receipt.Files = append(receipt.Files, SourceFile{Path: name, Bytes: int64(len(data)), SHA256: fmt.Sprintf("%x", sha256.Sum256(data))})
		}
		return root, receipt
	}
	t.Run("canonical relocation", func(t *testing.T) {
		first, a := makeSource(t)
		second, b := makeSource(t)
		slices.Reverse(b.Files)
		a, err := a.Verify(t.Context(), first)
		if err != nil {
			t.Fatal(err)
		}
		b, err = b.Verify(t.Context(), second)
		if err != nil {
			t.Fatal(err)
		}
		left, _ := json.Marshal(a)
		right, _ := json.Marshal(b)
		if string(left) != string(right) {
			t.Fatal("location or enumeration order changed receipt")
		}
		if a.Files[0].Path != "config.json" {
			t.Fatal("receipt is not sorted")
		}
	})
	for _, failure := range []string{"changed bytes", "missing file", "wrong size", "invalid digest", "duplicate", "escape", "absolute", "backslash", "mutable revision", "empty source", "empty files", "cancelled"} {
		t.Run(failure, func(t *testing.T) {
			root, receipt := makeSource(t)
			ctx := t.Context()
			switch failure {
			case "changed bytes":
				if err := os.WriteFile(filepath.Join(root, receipt.Files[0].Path), []byte("WEIGHTS.BIN"), 0600); err != nil {
					t.Fatal(err)
				}
			case "missing file":
				if err := os.Remove(filepath.Join(root, receipt.Files[0].Path)); err != nil {
					t.Fatal(err)
				}
			case "wrong size":
				receipt.Files[0].Bytes++
			case "invalid digest":
				receipt.Files[0].SHA256 = "not-a-sha256"
			case "duplicate":
				receipt.Files = append(receipt.Files, receipt.Files[0])
			case "escape":
				receipt.Files[0].Path = "../weights.bin"
			case "absolute":
				receipt.Files[0].Path = filepath.Join(root, "weights.bin")
			case "backslash":
				receipt.Files[0].Path = `component\weights.bin`
			case "mutable revision":
				receipt.Revision = "master"
			case "empty source":
				receipt.Source = ""
			case "empty files":
				receipt.Files = nil
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			got, err := receipt.Verify(ctx, root)
			if err == nil || len(got.Files) != 0 {
				t.Fatal("invalid source returned a verified receipt")
			}
			if failure == "cancelled" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost: %v", err)
			}
		})
	}
}
