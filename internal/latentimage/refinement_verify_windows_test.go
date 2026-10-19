//go:build windows

package latentimage

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMOSSRefinementVerificationRejectsChangedContract(t *testing.T) {
	for _, name := range []string{"checkpoint.json", "oracle-contract.json"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			for _, file := range []string{"checkpoint.json", "oracle-contract.json"} {
				data, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "moss", file))
				if err != nil {
					t.Fatal(err)
				}
				if file == name {
					data = append(data, ' ')
				}
				if err := os.WriteFile(filepath.Join(root, file), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			err := VerifyMOSSTransformer(t.Context(), "absent-checkpoint", "absent-reference", root, "refinement", io.Discard)
			if err == nil || !strings.Contains(err.Error(), "identity differs") || !strings.Contains(err.Error(), name) {
				t.Fatalf("changed %s reached model/device execution: %v", name, err)
			}
		})
	}
}
