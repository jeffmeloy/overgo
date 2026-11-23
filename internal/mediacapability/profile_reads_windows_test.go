//go:build windows

package mediacapability

import (
	"fmt"
	"testing"

	"overgo/internal/recipe"
)

// TestARepeatedPictureReadsNoProfile holds served-answers/profile (SQA-461): through the picture capability's serving
// (pictureExecutor, standInPictures), the repeat of a fixed-seed picture naming no preset, canvas or negative is
// answered by its stored run and reads the selected profile not at all, and the repeat of one naming a preset reads it
// once, for its naming and its sizing alike (catalogOnce). pictureFiles and preparedFirst had each read and parsed the
// profile before the stored answer was sought, a plain repeat asking neither for anything.
func TestARepeatedPictureReadsNoProfile(t *testing.T) {
	pictures := newStandInPictures(t, enhancerOpener)
	profile, named := pictures.execution.Program.Definition().PrimaryDependency(recipe.DependencyProfile)
	if !named {
		t.Fatal("the stand-in's recipe names no profile")
	}
	repeat := func(raw string) (reads int) {
		t.Helper()
		first, _ := pictures.served(t.Context(), raw)
		before := pictures.store.reads[profile]
		again, _ := pictures.served(t.Context(), raw)
		if again.Run != first.Run {
			t.Fatalf("%s was not answered by its stored run (%s, the first %s)", raw, again.Run, first.Run)
		}
		return pictures.store.reads[profile] - before
	}
	if reads := repeat(`{"prompt":"a lighthouse","seed":9,"width":256,"height":144}`); reads != 0 {
		t.Errorf("a plain picture's repeat read its profile %d times; want none", reads)
	}
	preset := fmt.Sprintf(`{"prompt":"a lighthouse at dusk","preset":"ultra-image-to-image-turbo","seed":11,"image":%q}`, pictures.sourceA.String())
	if reads := repeat(preset); reads != 1 {
		t.Errorf("a preset picture's repeat read its profile %d times; want once, for its naming and its sizing", reads)
	}
}
