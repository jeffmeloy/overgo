//go:build windows

package driver

import (
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

const (
	// nvmlTemperatureGPU is NVML_TEMPERATURE_GPU, the card's die sensor.
	nvmlTemperatureGPU = 0
	// nvmlPCIeReceived is NVML_PCIE_UTIL_RX_BYTES, the bytes the card receives over its link.
	nvmlPCIeReceived = 1
	// wattsPerMilliwatt scales NVML's power readings, which are milliwatts.
	wattsPerMilliwatt = 1e-3
	// bytesPerNVMLKilobyte scales NVML's PCIe throughput, which it reports in KB/s of 1024 bytes.
	bytesPerNVMLKilobyte = 1024
)

// cardQuery is one NVML query of a card: the function takes the card's
// handle, the arguments named here, and the address it answers through;
// scale turns the answer's first word into the reading's unit.
type cardQuery struct {
	reading, function string
	arguments         []uintptr
	scale             float64
}

var cardQueries = []cardQuery{
	{"utilization_percent", "nvmlDeviceGetUtilizationRates", nil, 1},
	{"temperature_celsius", "nvmlDeviceGetTemperature", []uintptr{nvmlTemperatureGPU}, 1},
	{"power_watts", "nvmlDeviceGetPowerUsage", nil, wattsPerMilliwatt},
	{"power_limit_watts", "nvmlDeviceGetEnforcedPowerLimit", nil, wattsPerMilliwatt},
	{"pcie_generation", "nvmlDeviceGetCurrPcieLinkGeneration", nil, 1},
	{"pcie_width", "nvmlDeviceGetCurrPcieLinkWidth", nil, 1},
	// The card samples its link for 20 ms to answer.
	{"pcie_received_bytes_per_second", "nvmlDeviceGetPcieThroughput", []uintptr{nvmlPCIeReceived}, bytesPerNVMLKilobyte},
}

// nvml is the NVIDIA Management Library the display driver installs beside
// nvcuda.dll, started once for the process's life; nil when it is absent or
// does not start.
var nvml = sync.OnceValue(func() *syscall.DLL {
	dll, err := syscall.LoadDLL("nvml.dll")
	if err != nil {
		return nil
	}
	if !nvmlCall(dll, "nvmlInit_v2") {
		_ = dll.Release()
		return nil
	}
	return dll
})

// nvmlCall reports whether an NVML function answered NVML_SUCCESS.
func nvmlCall(dll *syscall.DLL, function string, arguments ...uintptr) bool {
	proc, err := dll.FindProc(function)
	if err != nil {
		return false
	}
	result, _, _ := proc.Call(arguments...)
	return result == 0
}

// CardReadings reads the live load of the card uuid names (as DeviceUUID
// answers it) from NVML. A reading NVML refuses is left out, and none is
// read without NVML, because a card's load is optional.
func CardReadings(uuid string) []CardReading {
	dll := nvml()
	name, err := syscall.BytePtrFromString(uuid)
	if dll == nil || err != nil {
		return nil
	}
	var card uintptr
	// nvmlUtilization_t, the widest answer, leads with the GPU's share.
	var answer struct{ gpu, memory uint32 }
	var pinned runtime.Pinner
	defer pinned.Unpin()
	pinned.Pin(name)
	pinned.Pin(&card)
	pinned.Pin(&answer)
	if !nvmlCall(dll, "nvmlDeviceGetHandleByUUID", uintptr(unsafe.Pointer(name)), uintptr(unsafe.Pointer(&card))) {
		return nil
	}
	var readings []CardReading
	for _, query := range cardQueries {
		if nvmlCall(dll, query.function, append(append([]uintptr{card}, query.arguments...), uintptr(unsafe.Pointer(&answer)))...) {
			readings = append(readings, CardReading{query.reading, float64(answer.gpu) * query.scale})
		}
	}
	return readings
}
