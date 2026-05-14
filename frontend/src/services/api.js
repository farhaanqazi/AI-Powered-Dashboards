import axios from 'axios';

// Using relative paths for proxy in vite.config.js
const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
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

export const getDashboardData = async () => {
  const response = await api.get('/dashboard');
  return response.data;
};

export default api;