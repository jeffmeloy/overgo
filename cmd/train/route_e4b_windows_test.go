package main

import "overgo/internal/e4btrain"

// The e4b route publishes its trained model in its GGUF base's layout: its
// trainer names every tensor it trains as the artifact does, or the route
// would fall back to a trained-tensors-only weights file.
var _ servable = (*e4btrain.Trainer)(nil)
