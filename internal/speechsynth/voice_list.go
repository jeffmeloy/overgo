package speechsynth

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ListVoices names every voice the model directory exports, newest export
// generation first and each name once: the choices a request's voice
// control can offer, read from the artifact rather than typed anywhere.
func ListVoices(directory string) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	var generations []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "embeddings") {
			generations = append(generations, entry.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(generations)))
	seen := map[string]bool{}
	var voices []string
	for _, generation := range generations {
		exports, err := os.ReadDir(filepath.Join(directory, generation))
		if err != nil {
			return nil, err
		}
		for _, export := range exports {
			name, ok := strings.CutSuffix(export.Name(), ".safetensors")
			if export.IsDir() || !ok || seen[name] {
				continue
			}
			seen[name] = true
			voices = append(voices, name)
		}
	}
	return voices, nil
}
