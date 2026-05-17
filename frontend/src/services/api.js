import axios from 'axios';

const API_BASE_URL = '/api';

async function getClerkToken() {
  try {
    return (await window.Clerk?.session?.getToken()) || null;
  } catch {
    return null;
  }
}

function isGuestMode() {
  try { return localStorage.getItem('dataInsight:guestMode') === '1'; }
  catch { return false; }
}

function guestSessionId() {
  try { return localStorage.getItem('dataInsight:guestSessionId') || ''; }
  catch { return ''; }
}

async function authHeaders() {
  const token = await getClerkToken();
  if (token) return { Authorization: `Bearer ${token}` };
  if (isGuestMode()) {
    return {
      'X-Guest-Mode': '1',
      'X-Guest-Session-Id': guestSessionId(),
    };
  }
  return {};
}

const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use(async (config) => {
  const headers = await authHeaders();
  config.headers = { ...config.headers, ...headers };
  return config;
});

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('dataset', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const uploadFileStream = async (file, onPhase, { signal } = {}) => {
  const formData = new FormData();
  formData.append('dataset', file);

  const response = await fetch(`${API_BASE_URL}/upload/stream`, {
    method: 'POST',
    body: formData,
    headers: await authHeaders(),
    signal,
  });

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '');
    throw new Error(text || `Upload failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalPayload = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIdx;
    while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sepIdx);
      buffer = buffer.slice(sepIdx + 2);

      const dataLines = frame
        .split('\n')
        .filter(l => l.startsWith('data:'))
        .map(l => l.slice(5).trimStart());
      if (dataLines.length === 0) continue;

      let evt;
      try {
        evt = JSON.parse(dataLines.join('\n'));
      } catch (err) {
        console.warn('Failed to parse SSE frame', err, dataLines);
        continue;
      }

      if (typeof onPhase === 'function') onPhase(evt);

      if (evt.phase === 'error') {
        throw new Error(evt.message || 'Pipeline error');
      }
      if (evt.phase === 'done') {
        finalPayload = evt;
      }
    }
  }

  if (!finalPayload) {
    throw new Error('Upload stream ended without a final event.');
  }
  return finalPayload;
};

export const loadExternalSource = async (source) => {
  const response = await api.post('/load_external', {
    external_source: source,
  });

  return response.data;
};

export const validateExternalSource = async (source) => {
  const response = await api.post('/validate_external', { external_source: source });
  return response.data;
};

export const sniffCsvFile = async (file) => {
  if (!file) return { ok: false, error: 'No file provided.' };
  if (file.size === 0) return { ok: false, error: 'File is empty.' };
  if (!/\.csv$/i.test(file.name)) return { ok: false, error: 'File must have a .csv extension.' };

  const head = await file.slice(0, 8192).text();
  const stripped = head.trimStart();
  if (/^<!DOCTYPE|^<html|^<HTML|^<\?xml/.test(stripped)) {
    return { ok: false, error: 'File looks like HTML or XML, not CSV.' };
  }
  if (head.includes('\x00')) {
    return { ok: false, error: 'File appears to be binary, not text CSV.' };
  }

  const lines = head.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 2) {
    return { ok: false, error: 'File needs at least a header row and one data row.' };
  }

  const delimiters = [',', '\t', ';', '|'];
  const best = delimiters
    .map((d) => ({ d, c: lines[0].split(d).length }))
    .sort((a, b) => b.c - a.c)[0];
  if (best.c < 2) {
    return { ok: false, error: 'No CSV delimiter (comma, tab, or semicolon) found in header row.' };
  }

  const headerCount = best.c;
  const dataCount = lines[1].split(best.d).length;
  if (Math.abs(dataCount - headerCount) > Math.max(2, Math.floor(headerCount * 0.3))) {
    return {
      ok: false,
      error: `Header has ${headerCount} columns but row 2 has ${dataCount}. File may not be well-formed CSV.`,
    };
  }

  return { ok: true };
};

export const getDashboardData = async () => {
  const response = await api.get('/dashboard');
  return response.data;
};

// Phase 7 (S7.5): submit human schema-review overrides. The backend keys the
// dashboard by session, so the trace id is only a guard — 'current' is fine
// when the GET payload doesn't carry one.
export const patchRegistry = async (traceId, overrides) => {
  const response = await api.patch(
    `/dashboard/${encodeURIComponent(traceId || 'current')}/registry`,
    { overrides: overrides || [], confirm: true },
  );
  return response.data;
};

export default api;
