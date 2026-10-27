package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's native router top-10 (kernels/cuda/strata_native_router.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/native_router.cu): native_router_top10 and native_router_top10_multi, their host sides translated.

// The router's compiled geometry: Flash-Next's 512 experts, ten picks a token, a warp a token in blocks of eight warps.
const (
	strataRouterExperts    = 512
	strataRouterPicks      = 10
	strataRouterWarp       = 32
	strataRouterWarpsBlock = 8
)

// launchStrataNativeRoute is native_router_top10: one token's 512 router logits to its ten experts and weights.
func launchStrataNativeRoute(state *device.State, functions functionSet, logits, ids, weights driver.DevicePtr) error {
	if logits == 0 || ids == 0 || weights == 0 {
		return fmt.Errorf("native router requires aligned spans and disjoint outputs")
	}
	return launchGridABI(state, functions[kernelStrataNativeRoute], driver.Dim3{X: 1, Y: 1, Z: 1},
		driver.Dim3{X: strataRouterWarp, Y: strataRouterWarpsBlock, Z: 1}, &logits, &ids, &weights)
}

// launchStrataNativeRouteMulti is native_router_top10_multi: tokens rows of 512 logits to their ten experts and weights,
// a warp a token.
func launchStrataNativeRouteMulti(state *device.State, functions functionSet, logits, ids, weights driver.DevicePtr, tokens int32) error {
	if tokens < 1 || logits == 0 || ids == 0 || weights == 0 {
		return fmt.Errorf("native router (multi) requires aligned [n,512]/[n,10] buffers")
	}
	return launchGridABI(state, functions[kernelStrataNativeRouteMulti],
		driver.Dim3{X: (uint32(tokens) + strataRouterWarpsBlock - 1) / strataRouterWarpsBlock, Y: 1, Z: 1},
		driver.Dim3{X: strataRouterWarp, Y: strataRouterWarpsBlock, Z: 1}, &logits, &ids, &weights, &tokens)
}
