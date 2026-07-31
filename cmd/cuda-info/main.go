package main

import (
	"encoding/json"
	"fmt"
	"os"

	"llamacpp2go/internal/cuda/driver"
)

type report struct {
	DriverVersion string              `json:"driverVersion"`
	Devices       []driver.DeviceInfo `json:"devices"`
}

func run() error {
	lib, err := driver.Open()
	if err != nil {
		return err
	}
	defer lib.Close()

	if err := lib.Init(); err != nil {
		return err
	}
	version, err := lib.DriverVersion()
	if err != nil {
		return err
	}
	count, err := lib.DeviceCount()
	if err != nil {
		return err
	}

	result := report{
		DriverVersion: version.String(),
		Devices:       make([]driver.DeviceInfo, 0, count),
	}
	for ordinal := 0; ordinal < count; ordinal++ {
		info, infoErr := lib.DeviceInfo(ordinal)
		if infoErr != nil {
			return infoErr
		}
		result.Devices = append(result.Devices, info)
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cuda-info:", err)
		os.Exit(1)
	}
}
