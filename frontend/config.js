// Frontend runtime configuration
//
// NOTE FOR SINGLE-ORIGIN DEPLOYMENT:
// The recommended approach is to inline this directly in index.html:
//   <script>window.BACKEND_URL = window.location.origin || "";</script>
// The backend now serves static files at /static/ and the API at /,
// so no BACKEND_URL override is needed when the frontend is served by the backend.
//
// For a SEPARATE backend host (dual-origin), set it explicitly:
//   window.BACKEND_URL = "https://your-backend-host.example.com";
//
// The empty-string fallback in app.js is http://localhost:8000 —
// do NOT set BACKEND_URL = "" when config.js is loaded as a separate script,
// or the browser will try http://localhost:8000.
window.BACKEND_URL = "";
