package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

const imageVideoCompositionSHA256 = "6974a8c383d7a64cc0a02d79b41d2303ccea7760e0fa097572e013209e26461a"

func TestImageVideoCompositionRejectsChangedEvidence(t *testing.T) {
	t.Parallel()
	path := filepath.Join(testutil.RepoRoot(t), "docs/image_video_composition.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"cells", "components", "retained_run", "native_acquisition"} {
		t.Run(field, func(t *testing.T) {
			var changed map[string]any
			if err := json.Unmarshal(raw, &changed); err != nil {
				t.Fatal(err)
			}
			delete(changed, field)
			data, err := json.Marshal(changed)
			if err != nil {
				t.Fatal(err)
			}
			if err := checkMediaProtocolIdentity(data, imageVideoCompositionSHA256); err == nil {
				t.Fatal("composition evidence omission accepted")
			}
		})
	}
}
