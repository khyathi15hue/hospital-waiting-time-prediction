/*
 * auth.js -- drop-in staff login for doctor_dashboard.html
 *
 * Usage: add this as the FIRST script in the <head> of doctor_dashboard.html:
 *     <script src="auth.js"></script>
 *
 * What it does, without needing to know how the rest of the page is written:
 *   - shows a "Staff Sign In" screen until the user logs in
 *   - attaches "Authorization: Bearer <token>" to every request sent to the API
 *   - if the server answers 401 (missing/expired token), returns to the sign-in screen
 *   - turns the sidebar "Logout" item into a real logout
 *   - replaces the hardcoded "Dr. Smith" with the logged-in username
 */
(
 function () {
  const API_BASE = "http://127.0.0.1:8000";
  const TOKEN_KEY = "access_token";
  const USER_KEY = "doctor_username";

  const getToken = () => sessionStorage.getItem(TOKEN_KEY);

  // ---------- fetch wrapper: add token, catch 401 ----------
  const originalFetch = window.fetch.bind(window);

  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : input.url;
    const isApi = url.startsWith(API_BASE);
    const isLogin = url.startsWith(API_BASE + "/login");

    if (isApi && !isLogin && getToken()) {
      init = Object.assign({}, init);
      init.headers = Object.assign({}, init.headers, {
        Authorization: "Bearer " + getToken(),
      });
    }

    const res = await originalFetch(input, init);

    if (isApi && !isLogin && res.status === 401) {
      const hadSession = !!getToken();
      sessionStorage.removeItem(TOKEN_KEY);
      showLogin(hadSession ? "Session expired. Please sign in again." : "");
    }
    return res;
  };

  // ---------- login overlay ----------
  let overlay = null;

  function buildOverlay() {
    overlay = document.createElement("div");
    overlay.id = "authOverlay";
    overlay.style.cssText =
      "position:fixed;inset:0;z-index:99999;background:#F4F7FB;display:none;" +
      "align-items:center;justify-content:center;font-family:Inter,system-ui,sans-serif;";
    overlay.innerHTML = `
      <div style="background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:34px 32px;width:320px;
                  box-shadow:0 1px 2px rgba(15,42,74,.04),0 8px 24px rgba(15,42,74,.06);color:#0F2A4A;">
        <div style="font-size:18px;font-weight:700;margin-bottom:4px;">Staff Sign In</div>
        <div style="font-size:13px;color:#64748B;margin-bottom:20px;">CareWell Hospital — doctors and reception only</div>
        <label style="display:block;font-size:12.5px;color:#64748B;margin-bottom:5px;">Username</label>
        <input id="authUser" autocomplete="username"
               style="width:100%;box-sizing:border-box;padding:9px 11px;font-size:13.5px;border:1px solid #E2E8F0;border-radius:8px;margin-bottom:14px;" />
        <label style="display:block;font-size:12.5px;color:#64748B;margin-bottom:5px;">Password</label>
        <input id="authPass" type="password" autocomplete="current-password"
               style="width:100%;box-sizing:border-box;padding:9px 11px;font-size:13.5px;border:1px solid #E2E8F0;border-radius:8px;margin-bottom:16px;" />
        <button id="authBtn"
                style="width:100%;padding:11px;font-size:14px;font-weight:700;background:#2563EB;color:#fff;border:none;border-radius:8px;cursor:pointer;">
          Sign in
        </button>
        <div id="authErr" style="font-size:12.5px;color:#DC2626;margin-top:10px;min-height:16px;"></div>
      </div>`;
    document.body.appendChild(overlay);

    overlay.querySelector("#authBtn").addEventListener("click", doLogin);
    overlay.querySelector("#authPass").addEventListener("keydown", (e) => {
      if (e.key === "Enter") doLogin();
    });
  }

  function showLogin(message) {
    if (!overlay) buildOverlay();
    overlay.style.display = "flex";
    overlay.querySelector("#authErr").textContent = message || "";
    const user = overlay.querySelector("#authUser");
    if (!user.value) user.focus();
  }

  async function doLogin() {
    const username = overlay.querySelector("#authUser").value.trim();
    const password = overlay.querySelector("#authPass").value;
    const err = overlay.querySelector("#authErr");
    const btn = overlay.querySelector("#authBtn");

    if (!username || !password) {
      err.textContent = "Enter your username and password.";
      return;
    }
    btn.disabled = true;
    err.textContent = "";
    try {
      const res = await originalFetch(API_BASE + "/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ username, password }),
      });
      if (!res.ok) {
        err.textContent =
          res.status === 401 ? "Incorrect username or password." : "Login failed (server error " + res.status + ").";
        return;
      }
      const data = await res.json();
      sessionStorage.setItem(TOKEN_KEY, data.access_token);
      sessionStorage.setItem(USER_KEY, username);
      // Reload so the page's own startup code re-runs with a valid token
      window.location.reload();
    } catch (e) {
      err.textContent = "Can't reach the server at " + API_BASE + ". Is uvicorn running?";
    } finally {
      btn.disabled = false;
    }
  }

  // ---------- page wiring ----------
  document.addEventListener("DOMContentLoaded", function () {
    // Show the sign-in screen immediately if there's no session
    if (!getToken()) showLogin("");

    // Replace the hardcoded doctor name with the logged-in user
    const name = sessionStorage.getItem(USER_KEY);
    if (name && getToken()) {
      const nameEl = document.getElementById("docName");
      const initialEl = document.getElementById("docInitial");
      if (nameEl) nameEl.textContent = name;
      if (initialEl) initialEl.textContent = name.charAt(0).toUpperCase();
    }

    // Make the sidebar Logout item work
    document.addEventListener("click", function (e) {
      if (e.target.closest(".nav-item.logout")) {
        sessionStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem(USER_KEY);
        window.location.reload();
      }
    });
  });
})();