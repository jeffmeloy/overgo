package capabilityruntime

import (
	"crypto/rand"
	"runtime/debug"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// runKeyPrefix roots every capability run's workflow key.
const runKeyPrefix = "recipe/run/"

// codeScope names the code that answers requests in this process, which a
// replay is scoped to. An answer in the store is what some code gave, and a
// recipe's identity does not move when the code that computes it is mended,
// so an answer is given again only by the code that gave it. A binary built
// from a commit with nothing changed beyond it carries that commit, and every
// process of the build shares its answers. Any other build, from a changed
// tree, under go run, as a test, names only itself: it replays what this
// process answered and nothing older.
var codeScope = sync.OnceValue(func() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		revision, changed := "", true
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				revision = setting.Value
			case "vcs.modified":
				changed = setting.Value != "false"
			}
		}
		if revision != "" && !changed {
			return "commit/" + revision
		}
	}
	return "process/" + rand.Text()
})

// RunKey keys a capability run on the recipe, the exact request document and
// the code answering it, so an identical request to the same code replays the
// same run. A run a store kept from before a key named its code is keyed on
// the recipe and the request alone.
func RunKey(definition recipe.Definition, request artifact.Content) string {
	return runKeyPrefix + definition.ID.String() + "/" + request.Descriptor.ID.String() + "/" + codeScope()
}
