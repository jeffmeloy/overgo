package executor

import (
	"errors"

	"overgo/internal/cuda/driver"
)

// unloadModules unloads the executor's loaded kernel modules, a zero module skipped.
func unloadModules(lib *driver.Library, modules []driver.Module) error {
	var errs []error
	for _, module := range modules {
		if module != 0 {
			errs = append(errs, lib.ModuleUnload(module))
		}
	}
	return errors.Join(errs...)
}

// strataScoresTCSharedBytes is the size of the tensor-core block scorer's shared layout (kernels/cuda/strata_qsa.cu
// strata_block_scores_tc): a tile of 16 queries' four 128-wide indexer heads and a tile of 32 key blocks, each row
// padded by 4 floats against bank conflicts. It is past the default dynamic shared size, so the executor opts the
// kernel in when it loads its modules.
const strataScoresTCSharedBytes = uint32((16*(4*128+4) + 32*(128+4)) * 4)
