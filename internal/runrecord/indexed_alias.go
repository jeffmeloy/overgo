package runrecord

import (
	"fmt"

	"overgo/internal/artifact"
)

func indexedAlias(root string, identity artifact.ID, ordinal uint64) string {
	return root + identity.String() + "/" + fmt.Sprint(ordinal)
}
