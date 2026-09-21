package audioparity

import "overgo/internal/acceptancelane"

// These tests are a model acceptance suite, and the gate runs them after the
// commit with the lanes rather than in the phase that blocks it. Measured on
// 2026-09-21: the suite is about 335 s of whole-machine compute however it is
// scheduled -- 334 s at the default parallelism and 338 s capped at four on a
// 32 thread machine -- because its tests wait on child processes that run
// real speech models and repeat them to prove repeatability; the longest
// takes 93 s alone and 221 s beside the others. It was the critical path of
// every landing that touched a widely imported package, 6 min 45 s of about
// 8 min of blocking test wall.
var _ = acceptancelane.Declared
