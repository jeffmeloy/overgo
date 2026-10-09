package capabilityruntime

import (
	"crypto/rand"
	"reflect"
	"runtime/debug"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/inferencesurface"
	"overgo/internal/recipe"
)

// runKeyPrefix roots every capability run's workflow key.
const runKeyPrefix = "recipe/run/"

// codeScope names the code that answers requests in a process whose build
// counted no identity for a capability, which a replay is then scoped to. An
// answer in the store is what some code gave, and a recipe's identity does not
// move when the code that computes it is mended, so an answer is given again
// only by the code that gave it. A binary built from a commit with nothing
// changed beyond it carries that commit, and every process of the build
// shares its answers. Any other build, from a changed tree, under go run, as
// a test, names only itself: it replays what this process answered and
// nothing older.
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

// execution reads the identity this process keeps a package's answers under
// (inferencesurface.Execution); a test stands a build's count in for it.
var execution = inferencesurface.Execution

// answerScope names the code a capability's stored answer is given again by,
// for a capability whose model is a Model. The model's type says which
// package computes the answer, and a build that counted that package's
// execution identity keeps the answer under it: the sources, embedded files,
// kernels, modules and build settings the result depends on, with the
// libraries and settings of the process. A commit that changes none of them
// leaves the answer good, and one that changes any asks again. A build that
// counted none for the package keeps the scope of its code as a whole.
func answerScope[Model any]() string {
	model := reflect.TypeFor[Model]()
	if model.Kind() == reflect.Pointer {
		model = model.Elem()
	}
	if identity, held := execution(model.PkgPath()); held {
		return "identity/" + identity
	}
	return codeScope()
}

// RunKeyOf keys a capability run on the recipe, the exact request document and
// the code answering it for a model of type Model, so an identical request to
// the same code replays the same run. A run a store kept from before a key
// named its code is keyed on the recipe and the request alone.
func RunKeyOf[Model any](definition recipe.Definition, request artifact.Content) string {
	return runKeyPrefix + definition.ID.String() + "/" + request.Descriptor.ID.String() + "/" + answerScope[Model]()
}

// RunKey is the key of a run by a build that counted no identity, as every
// test's build is: the recipe, the request and the scope of the code as a
// whole. A capability that serves keys its runs with RunKeyOf.
func RunKey(definition recipe.Definition, request artifact.Content) string {
	return runKeyPrefix + definition.ID.String() + "/" + request.Descriptor.ID.String() + "/" + codeScope()
}
