package evaluation

import "overgo/internal/overgodb"

// gateDoor stands in for the gate in this package's fixtures: the store
// admits plan attempts from its capability alone.
var gateDoor = overgodb.NewProducer("gate")
