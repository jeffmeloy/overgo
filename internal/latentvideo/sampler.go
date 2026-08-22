package latentvideo

import "overgo/internal/sampling"

// UniPCSchedule exposes the shared shifted-flow schedule compiler.
var UniPCSchedule = sampling.UniPCSchedule

// UniPCSampler aliases the shared order-2 flow sampler.
type UniPCSampler = sampling.UniPCSampler

// NewUniPCSampler constructs the shared order-2 flow sampler.
var NewUniPCSampler = sampling.NewUniPCSampler

// GuideInto applies shared classifier-free guidance.
var GuideInto = sampling.GuideInto
