package main

import "overgo/internal/overgodb"

// gateDoor stands in for the gate in this package's fixtures: the store
// admits the gate's lifecycle records from its capability alone.
var gateDoor = overgodb.NewProducer("gate")
