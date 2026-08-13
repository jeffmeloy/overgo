//go:build !windows

package main

import (
	"overgo/internal/capabilityruntime"
	"overgo/internal/oscillatorimage"
)

func imageCapability() capability {
	return capability{
		inventory: imageGenInventory,
		execute: capabilityruntime.JSONScalar[oscillatorimage.Request, *oscillatorimage.Model, oscillatorimage.Image](
			"image-gen", oscillatorimage.ValidateRequest,
			capabilityruntime.IgnoreInput[oscillatorimage.Request](oscillatorimage.Load), oscillatorimage.RegisterRuntime,
		),
	}
}
