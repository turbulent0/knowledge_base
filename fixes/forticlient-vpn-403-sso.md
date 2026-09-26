# FortiClient SSL-VPN 403 on portal (eva_eu)

**Symptom:** `403 Forbidden` on `/sslvpn/portal/index.html` after SSO. Gateway `vpneu.eva.ua:10443` is reachable; first SAML succeeds, then a second SAML hits portal → 403.

**Cause:** Stale FortiClient/Microsoft SSO session; FortiClient retries SAML ~4s after success.

**Fix:**

1. Quit FortiClient fully.
2. Clear session cache:
  ```bash
   rm -f ~/.config/FortiClient/Cookies
   rm -rf ~/.config/FortiClient/"Session Storage"/*
   rm -rf ~/.config/FortiClient/"Local Storage"/leveldb/*
  ```
3. Reconnect once; do not click Connect again while first attempt runs.

**If still fails:** ask IT to check SSL-VPN portal assignment on FortiGate for your account.