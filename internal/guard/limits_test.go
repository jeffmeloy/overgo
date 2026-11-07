package guard

import "testing"

// TestTheGuardRefusesARawPushAndSecuritySettings holds two rules session memory held alone to the guard: a push goes
// only through the owner's command, which fast-forwards origin's main, so a raw git push is refused however its options
// are spelled, and fetching or pulling passes; and no session changes the machine's security settings, so Defender's
// command and reg, netsh and sc writing Defender's services, the firewall or UAC's policy key are refused, while reading
// a key or a service and writing an unrelated one pass.
func TestTheGuardRefusesARawPushAndSecuritySettings(t *testing.T) {
	t.Parallel()
	verdicts(t, map[string]bool{
		"git push origin master:main": true,
		"GCM_INTERACTIVE=Never git -c credential.helper=manager push origin 404b66d4d:refs/heads/main": true,
		"cd repo && git push --force origin HEAD":                                                      true,
		"go run ./cmd/loop -push ad389b059 -confirm-push":                                              false,
		"git fetch origin main":                                                                        false,
		"git pull --ff-only":                                                                           false,
		`reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System" /v EnableLUA /t REG_DWORD /d 0 /f`: true,
		`reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows Defender" /f`:                                          true,
		"netsh advfirewall set allprofiles state off":                                                                true,
		"sc stop WinDefend":                      true,
		"sc.exe config WdNisSvc start= disabled": true,
		`"C:\Program Files\Windows Defender\MpCmdRun.exe" -RemoveDefinitions -All`:   true,
		`reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"`: false,
		"sc query WinDefend": false,
		`reg add "HKCU\Software\overgo" /v Theme /d dark /f`: false,
	})
}
