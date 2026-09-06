package runrecord

// BackendRemote marks an environment whose model runs outside the store,
// at a hosted provider: nothing about such a run is reproducible from the
// store's bytes, and every record under the environment carries the mark.
const BackendRemote = "remote"

// Reproducible reports whether a run under this environment can be
// reproduced from the store; a run at a remote backend cannot.
func (e Environment) Reproducible() bool {
	return e.Backend != BackendRemote
}
